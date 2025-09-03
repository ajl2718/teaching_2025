# Easy Pandas: 
# Implements a simplified pandas dataframe for the purposes of 
# teaching. The operations filter, select, groupby, join, sort, add_column
# have been added / renamed to make it easier for students to learn about
# these data transformations while they are beginning to code in Python
# 
# Author: Alex Lee
# Date: 16 January 2025
import pandas as pd
from d2k_utils_2025 import *

def clean_numeric_string(text):
    return int(text.replace(',', ''))

class EasyDataFrame(pd.DataFrame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return EasyDataFrame

    def filter(self, column, predicate, value):
        """
        Filter a dataframe based on rows that meet a particular condition in a column

        Parameters
        ----------
        column (str): name of the column that is used to specify condition
        predicate (str): defines the condition used for filtering
                         Values include: 'equals', 'less than', 'greater than', 'within', 'in'
        value (str): the value that we are comparing to

        Returns
        -------
        df_new (EasyDataFrame): the filtered dataframe
        """
        if predicate == 'equals':
            df_new = self[self[column] == value]
            return df_new
        elif predicate == 'less than':
            df_new = self[self[column] < value]
            return df_new
        elif predicate == 'greater than':
            df_new = self[self[column] > value]
            return df_new
        elif predicate == 'within':
            if not isinstance(value, (tuple, list)) or len(value) != 2:
                raise ValueError("For 'within', value must be a tuple or list of length 2 (min, max).")
            df_new = self[self[column].between(value[0], value[1])]
            return df_new
        elif predicate == 'in':
            df_new = self[self[column].isin(value)]
            return df_new
        else:
            raise ValueError(f"Unsupported predicate: {predicate}")

    def select(self, columns):
        """
        Select a subset of the columns

        Parameters
        ----------
        columns (list): the names of the columns to select

        Returns
        -------
        df_new (EasyDataFrame): 
        """
        df_new = self.loc[:, columns]
        return df_new

    def groupby(self, groupby_column, value_column, agg_func):
        """
        Carry out a group by operation on the dataframe

        Parameters
        ----------
        groupby_column (str): name of the column to group by
        value_column (str): name of column to aggregate over
        agg_func (str): name of function to use in aggregation step
        
        Returns
        -------
        df_new (EasyDataFrame): grouped dataframe
        """
        # Define supported aggregation functions
        valid_agg_funcs = ['max', 'min', 'mean', 'median', 'first', 'sum']
        
        # Validate the aggregation function
        if agg_func not in valid_agg_funcs:
            raise ValueError(f"Invalid agg_func: {agg_func}. Supported functions are: {valid_agg_funcs}")
        
        # Perform groupby and aggregation
        grouped = super().groupby(groupby_column)[value_column].agg(agg_func)
        
        # Convert the result back to EasyDataFrame
        return EasyDataFrame(grouped.reset_index())
            
    def join(self, df2, left_on, right_on, how):
        """
        The usual pandas merge operation but renamed

        Parameters
        ----------
        df2 (EasyDataFrame): right-hand dataframe
        left_on (str): columns to use for joining in the left dataframe
        right_on (str): columns to use for joining in the right dataframe
        how (str): type of join

        Returns
        -------
        df_new (EasyDataFrame): the joined dataframe
        """
        df_new = self.merge(df2, left_on=left_on, right_on=right_on, how=how)
        return df_new

    def sort(self, column, ordering='ascending'):
        """
        Sort a dataframe based on a column

        Parameters
        ----------
        column (str): name of the column used to sort dataframe by
        ordering (str, default='ascending'): whether to sort in ascending or descending order

        Returns
        -------
        df_new (EasyDataFrame): the sorted dataframe
        """
        if ordering == 'ascending':
            df_new = self.sort_values(by=column, ascending=True)
        elif ordering == 'descending':
            df_new = self.sort_values(by=column, ascending=False)
        return df_new

    def add_column(self, column_name, values):
        """
        Add a new column to a dataframe with a specific value or set of values

        Parameters
        ----------
        column_name (str): name of the column to create
        values (str or list): values to set column to

        Returns
        -------
        df_new (EasyDataFrame): the dataframe with the column added
        """
        if column_name in self.columns:
            raise ValueError(f"{column_name} is already the name of a column in the dataframe. Use another name.")
            
        if isinstance(values, (str, int, float)):
            df_new = self.copy()
            df_new[column_name] = values
        elif isinstance(values, list):
            if len(values) != self.shape[0]:
                raise ValueError(f"Length of values ({len(values)}) does not match the number of rows in the dataframe ({self.shape[0]}).")
            df_new = self.copy()
            df_new[column_name] = values
        else:
            raise TypeError(f"Unsupported type for values: {type(values)}. Expected str, int, float, or list.")
            
        return df_new
    
    def add_derived_column(self, source_column, new_column, func):
        """
        Add a new column to a dataframe with a specific value or set of values

        Parameters
        ----------
        source_column (str): name of the column that new column derived from
        new_column (str): name of the new column to create
        func (function): function that is applied to the source column

        Returns
        -------
        df_new (EasyDataFrame): the dataframe with the column added
        """
        # Validate source_column existence
        if source_column not in self.columns:
            raise ValueError(f"The source column '{source_column}' does not exist in the dataframe.")
        
        # Apply the function to the source column and create the new column
        df_new = self.copy()
        try:
            df_new[new_column] = df_new[source_column].apply(func)
        except Exception as e:
            raise ValueError(f"Failed to apply the function to column '{source_column}': {e}")
        
        return df_new