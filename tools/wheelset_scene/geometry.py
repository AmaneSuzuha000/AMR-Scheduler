"""Small deterministic mesh builder and COLLADA writer (metres, Z up)."""
from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import xml.etree.ElementTree as ET


PALETTE = {
    "steel": (0.49, 0.53, 0.56, 1),
    "wheel_steel": (0.23, 0.25, 0.27, 1),
    "polished": (0.72, 0.76, 0.79, 1),
    "dark_steel": (0.19, 0.23, 0.25, 1),
    "gray": (0.53, 0.59, 0.61, 1),
    "machine": (0.33, 0.36, 0.38, 1),
    "light_gray": (0.79, 0.82, 0.82, 1),
    "orange": (0.91, 0.27, 0.035, 1),
    "orange_dark": (0.57, 0.14, 0.022, 1),
    "blue": (0.045, 0.38, 0.59, 1),
    "cyan": (0.03, 0.71, 0.69, 1),
    "black": (0.035, 0.045, 0.055, 1),
    "screen": (0.075, 0.12, 0.29, 1),
    "yellow": (0.96, 0.91, 0.045, 1),
    "green": (0.05, 0.72, 0.27, 1),
    "red": (0.78, 0.055, 0.025, 1),
    "floor": (0.42, 0.44, 0.45, 1),
    "wall": (0.31, 0.33, 0.34, 1),
    "spray_screen": (0.12, 0.14, 0.16, .38),
}


def add(a, b):
    return tuple(x + y for x, y in zip(a, b))


def sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def mul(a, k):
    return tuple(x * k for x in a)


def cross(a, b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])


def unit(a):
    length = math.sqrt(sum(x*x for x in a))
    if length < 1e-12:
        raise ValueError("Zero-length direction")
    return mul(a, 1 / length)


class Mesh:
    def __init__(self, name):
        self.name = name
        self.vertices = []
        self.faces = []
        self.materials = []
        self.parts = []

    def append(self, vertices, faces, material, part):
        offset = len(self.vertices)
        self.vertices.extend([tuple(float(x) for x in v) for v in vertices])
        self.faces.extend([tuple(offset+i for i in face) for face in faces])
        self.materials.extend([material] * len(faces))
        self.parts.extend([part] * len(faces))

    def box(self, center, size, material="gray", part="frame", yaw=0):
        x, y, z = [v / 2 for v in size]
        corners = [(-x,-y,-z),(x,-y,-z),(x,y,-z),(-x,y,-z),
                   (-x,-y,z),(x,-y,z),(x,y,z),(-x,y,z)]
        quads = [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]
        c, s = math.cos(yaw), math.sin(yaw)
        transformed = [add(center,(c*a-s*b,s*a+c*b,d)) for a,b,d in corners]
        for q in quads:
            self.append([transformed[i] for i in q], [(0,1,2),(0,2,3)], material, part)

    def beam(self, a, b, width, depth=None, material="gray", part="frame"):
        depth = width if depth is None else depth
        along = unit(sub(b,a))
        helper = (0,0,1) if abs(along[2]) < .95 else (0,1,0)
        u = unit(cross(along,helper)); v = cross(along,u)
        corners = [add(p, add(mul(u,sx*width/2),mul(v,sy*depth/2)))
                   for p in [a,b] for sx,sy in [(-1,-1),(1,-1),(1,1),(-1,1)]]
        for q in [(0,3,2,1),(4,5,6,7),(0,1,5,4),(1,2,6,5),(2,3,7,6),(3,0,4,7)]:
            self.append([corners[i] for i in q],[(0,1,2),(0,2,3)],material,part)

    def cylinder(self, a, b, radius, material="steel", part="shaft", segments=32):
        axis=unit(sub(b,a)); helper=(0,0,1) if abs(axis[2])<.95 else (0,1,0)
        u=unit(cross(axis,helper));v=cross(axis,u)
        rings=[[add(p,add(mul(u,radius*math.cos(2*math.pi*i/segments)),
                         mul(v,radius*math.sin(2*math.pi*i/segments)))) for i in range(segments)] for p in [a,b]]
        faces=[]
        for i in range(segments):
            j=(i+1)%segments
            faces.extend([(i,j,segments+j),(i,segments+j,segments+i)])
        self.append(rings[0]+rings[1],faces,material,part)
        for ring,point,flip in [(rings[0],a,True),(rings[1],b,False)]:
            faces=[(segments,(i+1)%segments,i) if flip else (segments,i,(i+1)%segments) for i in range(segments)]
            self.append(ring+[point],faces,material,part)

    def lathe_y(self, profile, center=(0,0,0), material="steel", part="wheel", segments=64):
        # Each profile band gets its own vertices: radial smoothing without blurring shoulders.
        for (y0,r0),(y1,r1) in zip(profile,profile[1:]):
            rings=[[add(center,(r*math.cos(2*math.pi*i/segments),y,r*math.sin(2*math.pi*i/segments)))
                    for i in range(segments)] for y,r in [(y0,r0),(y1,r1)]]
            faces=[]
            for i in range(segments):
                j=(i+1)%segments
                faces.extend([(i,segments+i,segments+j),(i,segments+j,j)])
            self.append(rings[0]+rings[1],faces,material,part)

    def torus_y(self, center, major, minor, material="steel", part="bearing", segments=40):
        profile=[(minor*math.sin(2*math.pi*i/12),major+minor*math.cos(2*math.pi*i/12)) for i in range(13)]
        self.lathe_y(profile,center,material,part,segments)

    def normals(self):
        acc=[[0.,0.,0.] for _ in self.vertices]
        for a,b,c in self.faces:
            n=cross(sub(self.vertices[b],self.vertices[a]),sub(self.vertices[c],self.vertices[a]))
            for k in (a,b,c):
                for j in range(3):acc[k][j]+=n[j]
        result=[]
        for n in acc:
            length=math.sqrt(sum(v*v for v in n))
            result.append([v/length for v in n] if length>1e-14 else [0,0,1])
        return result

    def bounds(self):
        return [[min(v[i] for v in self.vertices) for i in range(3)],
                [max(v[i] for v in self.vertices) for i in range(3)]]

    def as_dict(self):
        return {"name":self.name,"vertices":self.vertices,"faces":self.faces,
                "materials":self.materials,"parts":self.parts,"bounds":self.bounds()}

    def write_dae(self,path:Path):
        def elem(parent,tag,text=None,**attributes):
            e=ET.SubElement(parent,tag,{k:str(v) for k,v in attributes.items()})
            if text is not None:e.text=str(text)
            return e
        root=ET.Element("COLLADA",xmlns="http://www.collada.org/2005/11/COLLADASchema",version="1.4.1")
        asset=elem(root,"asset");elem(asset,"created","2026-11-07T00:00:00Z");elem(asset,"modified","2026-11-07T00:00:00Z")
        elem(asset,"unit",name="meter",meter="1");elem(asset,"up_axis","Z_UP")
        effects=elem(root,"library_effects");materials=elem(root,"library_materials")
        used=sorted(set(self.materials))
        for name in used:
            effect=elem(effects,"effect",id=name+"-fx")
            phong=elem(elem(elem(effect,"profile_COMMON"),"technique",sid="common"),"phong")
            rgba=" ".join(str(x) for x in PALETTE[name])
            elem(elem(phong,"ambient"),"color",rgba);elem(elem(phong,"diffuse"),"color",rgba)
            spec=.45 if name in {"steel","polished"} else .13
            elem(elem(phong,"specular"),"color",f"{spec} {spec} {spec} 1")
            elem(elem(phong,"shininess"),"float","45" if name in {"steel","polished"} else "12")
            if PALETTE[name][3]<1:
                elem(elem(phong,"transparent",opaque="A_ONE"),"color",f"1 1 1 {PALETTE[name][3]}")
                elem(elem(phong,"transparency"),"float","1")
            elem(elem(materials,"material",id=name,name=name),"instance_effect",url="#"+name+"-fx")
        geo=elem(elem(root,"library_geometries"),"geometry",id="geometry",name=self.name)
        mesh=elem(geo,"mesh")
        for name,values in [("position",self.vertices),("normal",self.normals())]:
            source=elem(mesh,"source",id=name)
            elem(source,"float_array"," ".join(f"{x:.7g}" for row in values for x in row),id=name+"-array",count=len(values)*3)
            access=elem(elem(source,"technique_common"),"accessor",source="#"+name+"-array",count=len(values),stride=3)
            for axis in "XYZ":elem(access,"param",name=axis,type="float")
        elem(elem(mesh,"vertices",id="vertices"),"input",semantic="POSITION",source="#position")
        grouped=defaultdict(list)
        for face,mat in zip(self.faces,self.materials):grouped[mat].append(face)
        for mat,faces in grouped.items():
            triangles=elem(mesh,"triangles",material=mat,count=len(faces))
            elem(triangles,"input",semantic="VERTEX",source="#vertices",offset=0)
            elem(triangles,"input",semantic="NORMAL",source="#normal",offset=1)
            elem(triangles,"p"," ".join(str(i) for face in faces for vertex in face for i in [vertex,vertex]))
        visual=elem(elem(root,"library_visual_scenes"),"visual_scene",id="Scene",name="Scene")
        inst=elem(elem(visual,"node",id="model",name=self.name,type="NODE"),"instance_geometry",url="#geometry")
        binding=elem(elem(inst,"bind_material"),"technique_common")
        for mat in used:elem(binding,"instance_material",symbol=mat,target="#"+mat)
        elem(elem(root,"scene"),"instance_visual_scene",url="#Scene")
        ET.indent(root)
        path.parent.mkdir(parents=True,exist_ok=True)
        ET.ElementTree(root).write(path,encoding="utf-8",xml_declaration=True)
