from pathlib import Path

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element
from typing import Iterable

class XMLHandler:
    def __init__(self, xml_parser : ET, xml_path : str | Path):
        
        self.tree = xml_parser.parse(xml_path)
        self.root = self.tree.getroot()
    
    def get_root(self):
        return self.root 
    
    def get_element_value(self, element : Element):
        element_value = element.text if element is not None else None
        return element_value

    def find_element(self, parent_element : Element, element_name : str):
        return parent_element.find(element_name)

    def find_all_elements(self, parent_element : Element, element_name : str):
        return parent_element.findall(element_name)

    def find_and_get_value(self, parent_element : Element, element_name : str):
        element = self.find_element(parent_element, element_name)
        element_value = self.get_element_value(element)

        return element_value
    
    def get_values_from_parent(self, parent_element : Element, 
                                     all_children_elements_names : Iterable[str]):
        
        children_elements_values = {}

        for child_element_name in all_children_elements_names:
            child_element_value = self.find_and_get_value(parent_element, child_element_name)
            children_elements_values.update({child_element_name: child_element_value})
        
        return children_elements_values 
    
    def find_element_and_get_children_values(self, parent_element : Element, 
                                                   new_parent_element_name : str,
                                                   all_children_elements_names : Iterable[str]):
        
        new_parent_element = self.find_element(parent_element, new_parent_element_name)
        children_elements_values = self.get_values_from_parent(new_parent_element, all_children_elements_names)
        return new_parent_element, children_elements_values

    
    
    