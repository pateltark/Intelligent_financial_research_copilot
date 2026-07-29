from docling.document_converter import DocumentConverter

# 1. Provide the path to your image file (or a URL)
source = "D:\intelligent_financial_research_copilot\edgar_docs\Table-Set-up-4.png"

# 2. Initialize the converter and process the image
converter = DocumentConverter()
result = converter.convert(source)

# 3. Print the cleanly formatted extracted text to the console
print(result.document.export_to_markdown())