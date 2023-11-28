import xml.etree.ElementTree as et
import glob
from lxml import etree
from assets import xml_files


def merge_xml_files(files, main_file):
    xml_files = glob.glob(files + "/*.xml")

    files = glob.glob(files)

    base_file = et.parse(files[0])
    base_data = base_file.getroot()

    for file in files[1:]:
        file_data = et.parse(file).getroot()
        for item in file_data.find("worldbody"):
            base_data.find("worldbody").append(item)
        for item in file_data.find("actuator"):
            base_data.find("actuator").append(item)

    # formatted_file = etree.tostring(base_data, pretty_print=True, encoding="UTF-8", xml_declaration=True)

    # with open(r"C:\Users\Nites\survivalenv-master\assets\final_file.xml", "wb") as f:
    #     f.write(formatted_file)

    base_file.write(main_file)

merge_xml_files(r"C:\Users\Nites\survivalenv-master\assets\xml_files\*.xml", r"C:\Users\Nites\survivalenv-master\assets\final_file.xml")

