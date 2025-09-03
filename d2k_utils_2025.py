# useful functions for the D2k Course
import pandas as pd
import numpy as np
import geopandas as gpd

import matplotlib.pyplot as plt
import seaborn as sns
import folium
import branca.colormap as cm
import json

from datetime import datetime 

from os import listdir

colnames = ['data_typer', 
            'cancer_subsite', 
            'year', 
            'sex', 
            'age_group', 
            'num_cases',
            'age_specific_rate', 
            'age_rate_std_2001', 
            'age_rate_std_2024', 
            'blank1', 
            'icd10_subsite', 
            'blank2']

folder = '/Users/alexlee/Desktop/Data/work'
filename = '/Users/alexlee/Desktop/Data/work/D2K/cancer_stats_aihw.csv'

filename_abs = '/Users/alexlee/Desktop/Data/work/D2K/abs_young_people_counts.csv'
filename_sa2_cancer = '/Users/alexlee/Desktop/Data/work/D2K/bowel_cancer_diagnoses_sa2_with_rate.csv'


def get_month(date_input):
    """
    Extract the month name from a datetime or string input.
    """
    if isinstance(date_input, str):
        try:
            date_input = datetime.strptime(date_input, '%Y-%m-%d')  # Modify format if needed
        except ValueError:
            raise ValueError("String date must be in the format 'YYYY-MM-DD'.")
    
    if isinstance(date_input, datetime):
        return date_input.strftime('%B')  # Returns the full month name
    else:
        raise TypeError("Input must be a datetime object or a string in the format 'YYYY-MM-DD'.")


def get_year(date_input):
    """
    Extract the year from a datetime or string input.
    """
    if isinstance(date_input, str):
        try:
            date_input = datetime.strptime(date_input, '%Y-%m-%d')  # Modify format if needed
        except ValueError:
            raise ValueError("String date must be in the format 'YYYY-MM-DD'.")
    
    if isinstance(date_input, datetime):
        return date_input.year
    else:
        raise TypeError("Input must be a datetime object or a string in the format 'YYYY-MM-DD'.")

def get_day_of_week(date_input):
    """
    Extract the day of the week from a datetime or string input.
    Returns the day of the week as an integer (0=Monday, 6=Sunday).
    """
    if isinstance(date_input, str):
        try:
            date_input = datetime.strptime(date_input, '%Y-%m-%d')  # Modify format if needed
        except ValueError:
            raise ValueError("String date must be in the format 'YYYY-MM-DD'.")
    
    if isinstance(date_input, datetime):
        return date_input.weekday()  # Monday=0, Sunday=6
    else:
        raise TypeError("Input must be a datetime object or a string in the format 'YYYY-MM-DD'.")