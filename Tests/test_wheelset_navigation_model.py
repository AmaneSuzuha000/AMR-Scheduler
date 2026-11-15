import xml.etree.ElementTree as ET

import pytest

from tools.wheelset_scene.navigation_model import WHEELS, set_lateral_compliance


def test_contact_compliance_is_on_each_collision_not_an_ignored_link_extension():
    root = ET.fromstring('<sdf><model name="robot"/></sdf>')
    for name in WHEELS:
        link = ET.SubElement(root.find('model'), 'link', name=name+'_wheel_link')
        ET.SubElement(link, 'slip1').text = '99'  # An ignored URDF extension.
        ode = ET.SubElement(ET.SubElement(ET.SubElement(ET.SubElement(link, 'collision'),
                            'surface'), 'friction'), 'ode')
        ET.SubElement(ode, 'fdir1').text = '0 0 1'
    set_lateral_compliance(root, .1)
    assert [e.text for e in root.findall('.//collision/surface/friction/ode/slip1')] == ['0.1']*4
    assert [e.text for e in root.findall('.//link/slip1')] == ['99']*4


@pytest.mark.parametrize('value', [0., -1., float('inf'), float('nan')])
def test_compliance_requires_finite_positive_value(value):
    with pytest.raises(ValueError):
        set_lateral_compliance(ET.fromstring('<sdf><model/></sdf>'), value)
