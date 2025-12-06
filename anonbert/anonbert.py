from .anonymizepredict import AnonymizePredict
from pathlib import Path

def anonimze_file(input_path: Path | str, model_dir: Path | str = "Matela7/AnonBert-ENR"):
    anonymizer = AnonymizePredict(input_path, model_dir)
    anon_text_path, _ = anonymizer.anonymize_file()

def anonimze_with_fakefill(input_path: Path | str, model_dir: Path | str = "Matela7/AnonBert-ENR"):
    anonymizer = AnonymizePredict(input_path, model_dir)
    anon_text_path, _ = anonymizer.anonymize_file()
    anonymizer.anonimize_fakefill(anon_text_path)


def anonymize_file(input_path: Path | str, model_dir: Path | str = "Matela7/AnonBert-ENR"):
    return anonimze_file(input_path, model_dir)


def anonymize_with_fakefill(input_path: Path | str, model_dir: Path | str = "Matela7/AnonBert-ENR"):
    return anonimze_with_fakefill(input_path, model_dir)