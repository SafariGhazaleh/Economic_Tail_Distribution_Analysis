#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Ghazaleh Safari 

Integrated International Master- and PhD program in Mathematics 
"""

import numpy as np
import pandas as pd

""" Class for holding, preparing, and assessing a Leonfief model from world
    inpit-output data."""
class Leontief_Model():
    def __init__(self):
        """
        Constructor method.
        Generates Leontief matrices from world input-output table for 2018 
        (most recent one available). It assumes the data to be in the current
        working directory under the file name ICIO2021_2018.csv.
        Data can be downloaded from the OECD website:
            https://www.oecd.org/sti/ind/inter-country-input-output-tables.htm
            http://stats.oecd.org/wbos/fileview2.aspx?IDFile=59a3d7f2-3f23-40d5-95ca-48da84c0f861
        
        Returns
        -------
        None.

        """
        """ Read data"""
        self.df = pd.read_csv("ICIO2021_2018.csv")
    
        """ Acertain that table is correct"""
        assert (self.df.iloc[0:3195,0] == self.df.columns[1:3196]).all()
    
        """ Extract Input-Output-Tables from rows 0-3194, columns 1-3195"""
        self.io_table =  self.df.iloc[0:3195,1:3196]
        self.io_table.index = list(self.df.iloc[0:3195,0])
        self.io_matrix = np.asarray(self.io_table)

        """ Extract final demand from rows 0-3195, columns 3195-3597"""
        self.demand = self.df.iloc[0:3195,3196:3598].sum(axis=1, numeric_only=True)

        """ Extract output from rows 0-3195, column 3598"""
        self.output = self.df.iloc[0:3195,3598]

        """ Compute production utilization coefficient matrix A"""
        self.A = np.divide(self.io_matrix, np.asarray(self.output)) 
        self.A[np.isnan(self.A)] = 0

        """ Output identity 
                np.dot(self.A, self.output) + self.demand == self.output
            holds only approximately because of rounding errors etc.
        """
        
        """ Compute Leontief inverse and inverse Leontief inverse"""
        self.I_minus_A = np.identity(3195) - self.A
        self.leontief_inverse = np.linalg.inv(self.I_minus_A)
        
        """ Output identity
                np.dot(self.leontief_inverse, self.demand)
            holds only approximately because of rounding errors etc.
            Prive level estimation can be dome like so:
                costs_plus_markup = np.ones(3195)
                self.price_levels = np.dot(costs_plus_markup, self.leontief_inverse)
        """

        """ Compute total output and total final demand for comparison for computing shock size"""
        self.total_output = np.sum(self.output)
        self.total_final_demand = np.sum(self.demand)


    def shock(self, shock_type="Demand", shock_size=0.3, sample_size=300, repeat=100):
        """
        Method for assessing the direct and indirect effect of random supply 
        and demand shocks (hitting random sectors in random countries) on the 
        world economy.

        Parameters
        ----------
        shock_type : str, optional
            Type of shock, either "Demand" or "Supply". The default is "Demand".
        shock_size : float, optional
            Share by which supply or demand is decreased. The default is 0.3.
        sample_size : int, optional
            Number of random country-sectors to be sampled to assess the 
            expected effect of a random shock on the world economy. The 
            default is 300.
        repeat : int, optional
            Number of times the experiment should be repeated. The default is 100.

        Returns
        -------
        result_dict : dict
            Raw data and statistical measures on the expected shock effect.
            Contains:
                - Shock_effect_data: Raw data
                - Average
                - Standard_deviation
                - Median
                - Upper_5%_quantile
                - Upper_1%_quantile
        """
        
        if shock_type == "Demand":
            matrix = self.leontief_inverse
            vector = self.demand
            benchmark = self.total_output
        elif shock_type == "Supply":
            matrix = self.I_minus_A
            vector = self.output
            benchmark = self.total_final_demand
        else:
            assert False, "Unknown shock type {:s}".format(shock_type)

        shock_effects = []
        sectors = np.random.choice(np.arange(3195), replace=False, size=sample_size)

        for _ in range(repeat):
            for sec in sectors:
                mod_vector = vector.copy()
                mod_vector[sec] = mod_vector[sec] * (1 - shock_size)
                result = np.dot(matrix, mod_vector)
                effect = 1 - np.sum(result) / benchmark
                shock_effects.append(effect)

        result_dict = {
            "Shock_effect_data": shock_effects,
            "Average": np.mean(shock_effects),
            "Standard_deviation": np.std(shock_effects),
            "Median": np.quantile(shock_effects, 0.5),
            "Upper_5%_quantile": np.quantile(shock_effects, 0.95),
            "Upper_1%_quantile": np.quantile(shock_effects, 0.99)
        }
        
        return result_dict


if __name__ == '__main__':
    L = Leontief_Model()

    """Reproducible upper 1% quantile for demand shocks with different sizes"""
    shock_sizes = [0.3, 0.7, 1.0]
    results = {}

    for size in shock_sizes:
        demand_shock_effects = L.shock(shock_type="Demand", shock_size=size, sample_size=300, repeat=100)
        results[size] = demand_shock_effects["Upper_1%_quantile"]

    print("Reproducible upper 1% quantile for demand shocks:")
    print(results)
