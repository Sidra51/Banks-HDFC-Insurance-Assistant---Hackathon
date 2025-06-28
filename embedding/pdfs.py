import os

def print_folder_structure(root_dir, indent=""):
    for item in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, item)
        if os.path.isdir(path):
            print(f"{indent}📁 {item}/")
            print_folder_structure(path, indent + "    ")
        elif item.lower().endswith(".md"):
            print(f"{indent}📄 {item}")
        else:
            print(f"{indent}📄 {item}")

# Replace 'policies' with the path to your folder
print("Folder structure:")
print_folder_structure("Insurance Plans")

# import os
# import aspose.words as aw
# import re

# # Input and output directories
# source_dir = "Insurance Plans"
# destination_dir = "insurance plans md"

# # Function to remove image links from markdown
# def remove_images_from_md(md_path):
#     with open(md_path, 'r', encoding='utf-8') as file:
#         content = file.read()
#     # Remove Markdown image syntax: ![](...) or ![alt](...)
#     content = re.sub(r'!\[.*?\]\(.*?\)', '', content)
#     with open(md_path, 'w', encoding='utf-8') as file:
#         file.write(content)

# # Function to convert PDF to Markdown
# def convert_pdf_to_md(src_path, dest_path):
#     try:
#         doc = aw.Document(src_path)
#         doc.save(dest_path, aw.SaveFormat.MARKDOWN)
#         remove_images_from_md(dest_path)
#         print(f"Converted (no images): {src_path} -> {dest_path}")
#     except Exception as e:
#         print(f"Error: {src_path} -> {e}")

# # Walk through folders and convert
# for root, _, files in os.walk(source_dir):
#     for file in files:
#         if file.lower().endswith(".pdf"):
#             src_file_path = os.path.join(root, file)
#             rel_path = os.path.relpath(root, source_dir)
#             dest_folder = os.path.join(destination_dir, rel_path)
#             os.makedirs(dest_folder, exist_ok=True)
#             dest_file_path = os.path.join(dest_folder, file.replace(".pdf", ".md"))
#             convert_pdf_to_md(src_file_path, dest_file_path)
