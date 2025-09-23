from infrastructure.service.pdf_converter import PDFConverter

class ConverterPDFUseCase:
    def __init__(self):
        self.converter = PDFConverter()

    def executar(self, pdf_path:str):
        return self.converter.converter_para_imagens(pdf_path)