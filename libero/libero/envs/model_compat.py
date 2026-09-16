"""Preserve LIBERO-era MJCF defaults when loading with modern MuJoCo."""

import xml.etree.ElementTree as ET


def preserve_libero_defaults(xml):
    """Keep the historical single-contact default, respecting explicit flags.

    MuJoCo 3.8 enabled multiccd by default. Legacy LIBERO XML omits this
    setting, and enabling it changes expert grasp replay on both backends.
    https://mujoco.readthedocs.io/en/3.8.0/changelog.html
    """
    root = ET.fromstring(xml)
    option = root.find("option")
    if option is None:
        option = ET.SubElement(root, "option")
    flag = option.find("flag")
    if flag is None:
        flag = ET.SubElement(option, "flag")
    if "multiccd" not in flag.attrib:
        flag.set("multiccd", "disable")
    return ET.tostring(root, encoding="unicode")
