import struct, sys
from PIL import Image
def xwd_to_png(src, dst):
    d = open(src, "rb").read()
    hdr = struct.unpack(">25I", d[:100])
    hsz, ver, fmt, depth, w, h = hdr[0], hdr[1], hdr[2], hdr[3], hdr[4], hdr[5]
    byte_order, bpp, bpl = hdr[7], hdr[11], hdr[12]
    ncolors = hdr[19]
    off = hsz + ncolors * 12
    print("hdr", hsz, "ver", ver, "fmt", fmt, "depth", depth, w, h, "bpp", bpp, "bpl", bpl, "ncol", ncolors, "off", off, "len", len(d))
    im = Image.new("RGB", (w, h))
    px = im.load()
    if bpp == 32:
        for y in range(h):
            row = off + y * bpl
            for x in range(w):
                p = row + x * 4
                b0, b1, b2, _ = d[p], d[p+1], d[p+2], d[p+3]
                px[x, y] = (b2, b1, b0)
    elif bpp == 24:
        for y in range(h):
            row = off + y * bpl
            for x in range(w):
                p = row + x * 3
                b0, b1, b2 = d[p], d[p+1], d[p+2]
                px[x, y] = (b2, b1, b0)
    elif bpp == 16:
        for y in range(h):
            row = off + y * bpl
            for x in range(w):
                p = row + x * 2
                v = struct.unpack(">H", d[p:p+2])[0]
                px[x, y] = (((v >> 11) & 31) * 255 // 31, ((v >> 5) & 63) * 255 // 63, (v & 31) * 255 // 31)
    im.save(dst)
    print("saved", dst, im.size)
xwd_to_png(sys.argv[1], sys.argv[2])