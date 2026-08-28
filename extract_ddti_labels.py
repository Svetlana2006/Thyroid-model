import os
import glob
import xml.etree.ElementTree as ET
import kagglehub

path = kagglehub.dataset_download('dasmehdixtr/ddti-thyroid-ultrasound-images')
xml_files = glob.glob(os.path.join(path, "*.xml"))

tirads_counts = {}
for xml_file in xml_files:
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        tirads_elem = root.find("tirads")
        if tirads_elem is not None:
            tirads = tirads_elem.text
            tirads_counts[tirads] = tirads_counts.get(tirads, 0) + 1
        else:
            tirads_counts["Missing"] = tirads_counts.get("Missing", 0) + 1
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")

print("TIRADS Distribution:")
print(tirads_counts)
