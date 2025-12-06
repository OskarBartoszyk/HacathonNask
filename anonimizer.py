"""
Biblioteka anonimizer - narzędzie do anonimizacji danych osobowych w tekście
Używa biblioteki Faker do generowania fałszywych danych
"""

from faker import Faker
import re
from datetime import datetime, timedelta
import random


class Anonimizer:
    """Klasa do anonimizacji tekstu z tagami"""
    
    def __init__(self, locale='pl_PL'):
        """
        Inicjalizacja anonimizera
        
        Args:
            locale: Lokalizacja dla Faker (domyślnie 'pl_PL')
        """
        self.faker = Faker(locale)
        self.text = ""
        
    def ReadText(self, filename):
        """
        Wczytuje tekst z pliku
        
        Args:
            filename: Nazwa pliku do wczytania
            
        Returns:
            self: Zwraca obiekt dla łańcuchowego wywoływania metod
        """
        with open(filename, 'r', encoding='utf-8') as file:
            self.text = file.read()
        return self
    
    def _generate_pesel(self):
        """Generuje losowy numer PESEL"""
        # Uproszczona wersja - generuje 11 cyfr
        return ''.join([str(random.randint(0, 9)) for _ in range(11)])
    
    def _generate_document_number(self):
        """Generuje losowy numer dokumentu (np. dowodu osobistego)"""
        letters = ''.join([chr(random.randint(65, 90)) for _ in range(3)])
        numbers = ''.join([str(random.randint(0, 9)) for _ in range(6)])
        return f"{letters}{numbers}"
    
    def _generate_bank_account(self):
        """Generuje losowy numer konta bankowego (IBAN PL)"""
        numbers = ''.join([str(random.randint(0, 9)) for _ in range(26)])
        return f"PL{numbers}"
    
    def _generate_credit_card(self):
        """Generuje losowy numer karty kredytowej"""
        return self.faker.credit_card_number()
    
    def _generate_username(self):
        """Generuje losową nazwę użytkownika"""
        return self.faker.user_name()
    
    def _generate_secret(self):
        """Generuje losowy sekret/hasło"""
        return self.faker.password(length=12, special_chars=True, digits=True, upper_case=True, lower_case=True)
    
    def _generate_pii(self):
        """Generuje losowe dane osobowe (Personal Identifiable Information)"""
        return f"{self.faker.first_name()} {self.faker.last_name()}, {self.faker.email()}"
    
    def FakeFillAll(self):
        """
        Znajduje wszystkie tagi w tekście i zastępuje je fałszywymi danymi
        
        Returns:
            str: Tekst z wypełnionymi tagami
        """
        result = self.text
        
        # Mapowanie tagów na metody Faker
        tag_mapping = {
            r'\[name\]': lambda: self.faker.first_name(),
            r'\[surname\]': lambda: self.faker.last_name(),
            r'\[age\]': lambda: str(random.randint(18, 90)),
            r'\[date-of-birth\]': lambda: self.faker.date_of_birth(minimum_age=18, maximum_age=90).strftime('%Y-%m-%d'),
            r'\[date\]': lambda: self.faker.date_between(start_date='-5y', end_date='today').strftime('%Y-%m-%d'),
            r'\[sex\]': lambda: random.choice(['Mężczyzna', 'Kobieta', 'Inna']),
            r'\[religion\]': lambda: random.choice(['Katolicyzm', 'Protestantyzm', 'Islam', 'Judaizm', 'Buddyzm', 'Hinduizm', 'Ateizm', 'Agnostycyzm']),
            r'\[political-view\]': lambda: random.choice(['Lewicowe', 'Centrolewicowe', 'Centrowe', 'Centroprawicowe', 'Prawicowe', 'Libertariańskie', 'Apolityczne']),
            r'\[ethnicity\]': lambda: random.choice(['Polska', 'Ukraińska', 'Białoruska', 'Romska', 'Niemiecka', 'Rosyjska', 'Litewska']),
            r'\[sexual-orientation\]': lambda: random.choice(['Heteroseksualna', 'Homoseksualna', 'Biseksualna', 'Aseksualna', 'Panseksualna']),
            r'\[health\]': lambda: random.choice(['Dobry', 'Bardzo dobry', 'Średni', 'Wymaga leczenia', 'Chroniczna choroba']),
            r'\[relative\]': lambda: f"{self.faker.first_name()} {self.faker.last_name()}",
            r'\[city\]': lambda: self.faker.city(),
            r'\[address\]': lambda: self.faker.address().replace('\n', ', '),
            r'\[email\]': lambda: self.faker.email(),
            r'\[phone\]': lambda: self.faker.phone_number(),
            r'\[pesel\]': lambda: self._generate_pesel(),
            r'\[document-number\]': lambda: self._generate_document_number(),
            r'\[company\]': lambda: self.faker.company(),
            r'\[school-name\]': lambda: f"{random.choice(['Szkoła Podstawowa', 'Liceum', 'Technikum', 'Uniwersytet'])} {self.faker.last_name()}",
            r'\[job-title\]': lambda: self.faker.job(),
            r'\[bank-account\]': lambda: self._generate_bank_account(),
            r'\[credit-card-number\]': lambda: self._generate_credit_card(),
            r'\[username\]': lambda: self._generate_username(),
            r'\[secret\]': lambda: self._generate_secret(),
            r'\[pii\]': lambda: self._generate_pii(),
        }
        
        # Zastępowanie każdego wystąpienia tagu
        for pattern, generator in tag_mapping.items():
            # Znajdujemy wszystkie wystąpienia danego tagu
            while re.search(pattern, result):
                # Dla każdego wystąpienia generujemy nową wartość
                result = re.sub(pattern, generator(), result, count=1)
        
        return result


# Funkcja pomocnicza dla uproszczonego użycia
def ReadText(filename):
    """
    Pomocnicza funkcja do wczytywania i tworzenia obiektu Anonimizer
    
    Args:
        filename: Nazwa pliku do wczytania
        
    Returns:
        Anonimizer: Obiekt z wczytanym tekstem
    """
    anonimizer = Anonimizer()
    return anonimizer.ReadText(filename)