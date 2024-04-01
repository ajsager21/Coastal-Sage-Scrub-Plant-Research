#!/usr/bin/env python
# coding: utf-8

# In[39]:


#ARTCAL PLOTS


# In[44]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import poisson, bartlett

# Load datasets
leaf_cn = pd.read_csv("Leaf_CN_r.csv")
sla = pd.read_csv("SLA.csv")
herb_mass = pd.read_csv("Herb_mass.csv")

# Merge datasets based on the common column "Plant"
merged_data = pd.merge(leaf_cn, sla, on="Plant")
merged_data = pd.merge(merged_data, herb_mass, on="Plant")

# Filter data for the specific species 'Artemisia_californica'
species_data = merged_data[merged_data['Species'] == 'Artemisia_californica']

# Generate hypothetical count data for herbivores (e.g., herbivore sightings)
np.random.seed(0)
mean_herbivore_count = 5  # Adjust as needed
species_data['herbivore_count'] = np.random.poisson(mean_herbivore_count, size=len(species_data))

# Handle missing values
species_data.dropna(inplace=True)

# Define and fit the mixed effects model
model = smf.mixedlm("herbivore_count ~ Herbivore_mass_mg", species_data, groups=species_data["Plot_x"] + species_data["Plant"])
result = model.fit()

# Calculate the residuals
residuals = result.resid

# Calculate the theoretical quantiles
sorted_residuals = np.sort(residuals)
n = len(sorted_residuals)
theoretical_quantiles = sm.distributions.ECDF(sorted_residuals)(sorted_residuals)

# Fit a trend line
slope, intercept = np.polyfit(theoretical_quantiles, sorted_residuals, 1)
trendline = slope * theoretical_quantiles + intercept

# Perform Bartlett test for dispersion
bartlett_statistic, bartlett_p_value = bartlett(sorted_residuals, trendline)

# Plot the Q-Q plot with trend line and dispersion test results
plt.figure(figsize=(8, 6))
plt.scatter(theoretical_quantiles, sorted_residuals, color='blue', alpha=0.5, label='Residuals')
plt.plot(theoretical_quantiles, trendline, color='red', linestyle='--', label='Trend line')
plt.text(0.5, -3, f'Bartlett statistic: {bartlett_statistic:.2f}\nBartlett p-value: {bartlett_p_value:.4f}', fontsize=10, ha='center')
plt.title('Q-Q Plot of Residuals for Artemisia_californica with Trend Line and Dispersion Test')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Residuals')
plt.legend()
plt.grid(True)
plt.show()


# In[38]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import poisson, bartlett

# Load datasets
leaf_cn = pd.read_csv("Leaf_CN_r.csv")
sla = pd.read_csv("SLA.csv")
herb_mass = pd.read_csv("Herb_mass.csv")

# Merge datasets based on the common column "Plant"
merged_data = pd.merge(leaf_cn, sla, on="Plant")
merged_data = pd.merge(merged_data, herb_mass, on="Plant")

# Filter data for the specific species 'Artemisia_californica'
species_data = merged_data[merged_data['Species'] == 'Artemisia_californica']

# Generate hypothetical count data for herbivores (e.g., herbivore sightings)
np.random.seed(0)
mean_herbivore_count = 5  # Adjust as needed
species_data['herbivore_count'] = np.random.poisson(mean_herbivore_count, size=len(species_data))

# Handle missing values
species_data.dropna(inplace=True)

# Define and fit the mixed effects model for Perc_N
model_perc_n = smf.mixedlm("herbivore_count ~ Perc_N", species_data, groups=species_data["Plot_x"] + species_data["Plant"])
result_perc_n = model_perc_n.fit()

# Calculate the residuals for Perc_N
residuals_perc_n = result_perc_n.resid

# Calculate the theoretical quantiles for Perc_N
sorted_residuals_perc_n = np.sort(residuals_perc_n)
n_perc_n = len(sorted_residuals_perc_n)
theoretical_quantiles_perc_n = sm.distributions.ECDF(sorted_residuals_perc_n)(sorted_residuals_perc_n)

# Fit a trend line for Perc_N
slope_perc_n, intercept_perc_n = np.polyfit(theoretical_quantiles_perc_n, sorted_residuals_perc_n, 1)
trendline_perc_n = slope_perc_n * theoretical_quantiles_perc_n + intercept_perc_n

# Perform Bartlett test for dispersion for Perc_N
bartlett_statistic_perc_n, bartlett_p_value_perc_n = bartlett(sorted_residuals_perc_n, trendline_perc_n)

# Plot the Q-Q plot with trend line and dispersion test results for Perc_N
plt.figure(figsize=(8, 6))
plt.scatter(theoretical_quantiles_perc_n, sorted_residuals_perc_n, color='blue', alpha=0.5, label='Residuals Perc_N')
plt.plot(theoretical_quantiles_perc_n, trendline_perc_n, color='red', linestyle='--', label='Trend line Perc_N')
plt.text(0.5, -3, f'Bartlett statistic: {bartlett_statistic_perc_n:.2f}\nBartlett p-value: {bartlett_p_value_perc_n:.4f}', fontsize=10, ha='center')
plt.title('Q-Q Plot of Residuals for Artemisia_californica with Perc_N with Trend Line and Dispersion Test')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Residuals Perc_N')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import poisson, bartlett

# Load datasets
leaf_cn = pd.read_csv("Leaf_CN_r.csv")
sla = pd.read_csv("SLA.csv")
herb_mass = pd.read_csv("Herb_mass.csv")

# Merge datasets based on the common column "Plant"
merged_data = pd.merge(leaf_cn, sla, on="Plant")
merged_data = pd.merge(merged_data, herb_mass, on="Plant")

# Filter data for the specific species 'Artemisia_californica'
species_data = merged_data[merged_data['Species'] == 'Artemisia_californica']

# Generate hypothetical count data for herbivores (e.g., herbivore sightings)
np.random.seed(0)
mean_herbivore_count = 5  # Adjust as needed
species_data['herbivore_count'] = np.random.poisson(mean_herbivore_count, size=len(species_data))

# Handle missing values
species_data.dropna(inplace=True)

# Transform 'Perc_C' to its logarithm
species_data['log_Perc_C'] = np.log(species_data['Perc_C'])

# Define and fit the mixed effects model for log(Perc_C)
model_log_perc_c = smf.mixedlm("herbivore_count ~ log_Perc_C", species_data, groups=species_data["Plot_x"] + species_data["Plant"])
result_log_perc_c = model_log_perc_c.fit()

# Calculate the residuals for log(Perc_C)
residuals_log_perc_c = result_log_perc_c.resid

# Calculate the theoretical quantiles for log(Perc_C)
sorted_residuals_log_perc_c = np.sort(residuals_log_perc_c)
n_log_perc_c = len(sorted_residuals_log_perc_c)
theoretical_quantiles_log_perc_c = sm.distributions.ECDF(sorted_residuals_log_perc_c)(sorted_residuals_log_perc_c)

# Fit a trend line for log(Perc_C)
slope_log_perc_c, intercept_log_perc_c = np.polyfit(theoretical_quantiles_log_perc_c, sorted_residuals_log_perc_c, 1)
trendline_log_perc_c = slope_log_perc_c * theoretical_quantiles_log_perc_c + intercept_log_perc_c

# Perform Bartlett test for dispersion for log(Perc_C)
bartlett_statistic_log_perc_c, bartlett_p_value_log_perc_c = bartlett(sorted_residuals_log_perc_c, trendline_log_perc_c)

# Plot the Q-Q plot with trend line and dispersion test results for log(Perc_C)
plt.figure(figsize=(8, 6))
plt.scatter(theoretical_quantiles_log_perc_c, sorted_residuals_log_perc_c, color='blue', alpha=0.5, label='Residuals log(Perc_C)')
plt.plot(theoretical_quantiles_log_perc_c, trendline_log_perc_c, color='red', linestyle='--', label='Trend line log(Perc_C)')
plt.text(0.5, -3, f'Bartlett statistic: {bartlett_statistic_log_perc_c:.2f}\nBartlett p-value: {bartlett_p_value_log_perc_c:.4f}', fontsize=10, ha='center')
plt.title('Q-Q Plot of Residuals for Artemisia_californica with log(Perc_C) with Trend Line and Dispersion Test')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Residuals log(Perc_C)')
plt.legend()
plt.grid(True)
plt.show()


# In[40]:


#ENCAL PLOTS


# In[42]:


from scipy.stats import poisson, bartlett

# Load datasets
leaf_cn = pd.read_csv("Leaf_CN_r.csv")
sla = pd.read_csv("SLA.csv")
herb_mass = pd.read_csv("Herb_mass.csv")

# Merge datasets based on the common column "Plant"
merged_data = pd.merge(leaf_cn, sla, on="Plant")
merged_data = pd.merge(merged_data, herb_mass, on="Plant")

# Filter data for the specific species 'Artemisia_californica'
species_data = merged_data[merged_data['Species'] == 'Encelia_californica']

# Generate hypothetical count data for herbivores (e.g., herbivore sightings)
np.random.seed(0)
mean_herbivore_count = 5  # Adjust as needed
species_data['herbivore_count'] = np.random.poisson(mean_herbivore_count, size=len(species_data))

# Handle missing values
species_data.dropna(inplace=True)

# Define and fit the mixed effects model
model = smf.mixedlm("herbivore_count ~ Herbivore_mass_mg", species_data, groups=species_data["Plot_x"] + species_data["Plant"])
result = model.fit()

# Calculate the residuals
residuals = result.resid

# Calculate the theoretical quantiles
sorted_residuals = np.sort(residuals)
n = len(sorted_residuals)
theoretical_quantiles = sm.distributions.ECDF(sorted_residuals)(sorted_residuals)

# Fit a trend line
slope, intercept = np.polyfit(theoretical_quantiles, sorted_residuals, 1)
trendline = slope * theoretical_quantiles + intercept

# Perform Bartlett test for dispersion
bartlett_statistic, bartlett_p_value = bartlett(sorted_residuals, trendline)

# Plot the Q-Q plot with trend line and dispersion test results
plt.figure(figsize=(8, 6))
plt.scatter(theoretical_quantiles, sorted_residuals, color='blue', alpha=0.5, label='Residuals')
plt.plot(theoretical_quantiles, trendline, color='red', linestyle='--', label='Trend line')
plt.text(0.5, -3, f'Bartlett statistic: {bartlett_statistic:.2f}\nBartlett p-value: {bartlett_p_value:.4f}', fontsize=10, ha='center')
plt.title('Q-Q Plot of Residuals for Encelia_californica with Trend Line and Dispersion Test')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Residuals')
plt.legend()
plt.grid(True)
plt.show()


# In[45]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import poisson, bartlett

# Load datasets
leaf_cn = pd.read_csv("Leaf_CN_r.csv")
sla = pd.read_csv("SLA.csv")
herb_mass = pd.read_csv("Herb_mass.csv")

# Merge datasets based on the common column "Plant"
merged_data = pd.merge(leaf_cn, sla, on="Plant")
merged_data = pd.merge(merged_data, herb_mass, on="Plant")

# Filter data for the specific species 'Artemisia_californica'
species_data = merged_data[merged_data['Species'] == 'Encelia_californica']

# Generate hypothetical count data for herbivores (e.g., herbivore sightings)
np.random.seed(0)
mean_herbivore_count = 5  # Adjust as needed
species_data['herbivore_count'] = np.random.poisson(mean_herbivore_count, size=len(species_data))

# Handle missing values
species_data.dropna(inplace=True)

# Define and fit the mixed effects model for Perc_N
model_perc_n = smf.mixedlm("herbivore_count ~ Perc_N", species_data, groups=species_data["Plot_x"] + species_data["Plant"])
result_perc_n = model_perc_n.fit()

# Calculate the residuals for Perc_N
residuals_perc_n = result_perc_n.resid

# Calculate the theoretical quantiles for Perc_N
sorted_residuals_perc_n = np.sort(residuals_perc_n)
n_perc_n = len(sorted_residuals_perc_n)
theoretical_quantiles_perc_n = sm.distributions.ECDF(sorted_residuals_perc_n)(sorted_residuals_perc_n)

# Fit a trend line for Perc_N
slope_perc_n, intercept_perc_n = np.polyfit(theoretical_quantiles_perc_n, sorted_residuals_perc_n, 1)
trendline_perc_n = slope_perc_n * theoretical_quantiles_perc_n + intercept_perc_n

# Perform Bartlett test for dispersion for Perc_N
bartlett_statistic_perc_n, bartlett_p_value_perc_n = bartlett(sorted_residuals_perc_n, trendline_perc_n)

# Plot the Q-Q plot with trend line and dispersion test results for Perc_N
plt.figure(figsize=(8, 6))
plt.scatter(theoretical_quantiles_perc_n, sorted_residuals_perc_n, color='blue', alpha=0.5, label='Residuals Perc_N')
plt.plot(theoretical_quantiles_perc_n, trendline_perc_n, color='red', linestyle='--', label='Trend line Perc_N')
plt.text(0.5, -3, f'Bartlett statistic: {bartlett_statistic_perc_n:.2f}\nBartlett p-value: {bartlett_p_value_perc_n:.4f}', fontsize=10, ha='center')
plt.title('Q-Q Plot of Residuals for Encelia_californica with Perc_N with Trend Line and Dispersion Test')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Residuals Perc_N')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import poisson, bartlett

# Load datasets
leaf_cn = pd.read_csv("Leaf_CN_r.csv")
sla = pd.read_csv("SLA.csv")
herb_mass = pd.read_csv("Herb_mass.csv")

# Merge datasets based on the common column "Plant"
merged_data = pd.merge(leaf_cn, sla, on="Plant")
merged_data = pd.merge(merged_data, herb_mass, on="Plant")

# Filter data for the specific species 'Artemisia_californica'
species_data = merged_data[merged_data['Species'] == 'Encelia_californica']

# Generate hypothetical count data for herbivores (e.g., herbivore sightings)
np.random.seed(0)
mean_herbivore_count = 5  # Adjust as needed
species_data['herbivore_count'] = np.random.poisson(mean_herbivore_count, size=len(species_data))

# Handle missing values
species_data.dropna(inplace=True)

# Transform 'Perc_C' to its logarithm
species_data['log_Perc_C'] = np.log(species_data['Perc_C'])

# Define and fit the mixed effects model for log(Perc_C)
model_log_perc_c = smf.mixedlm("herbivore_count ~ log_Perc_C", species_data, groups=species_data["Plot_x"] + species_data["Plant"])
result_log_perc_c = model_log_perc_c.fit()

# Calculate the residuals for log(Perc_C)
residuals_log_perc_c = result_log_perc_c.resid

# Calculate the theoretical quantiles for log(Perc_C)
sorted_residuals_log_perc_c = np.sort(residuals_log_perc_c)
n_log_perc_c = len(sorted_residuals_log_perc_c)
theoretical_quantiles_log_perc_c = sm.distributions.ECDF(sorted_residuals_log_perc_c)(sorted_residuals_log_perc_c)

# Fit a trend line for log(Perc_C)
slope_log_perc_c, intercept_log_perc_c = np.polyfit(theoretical_quantiles_log_perc_c, sorted_residuals_log_perc_c, 1)
trendline_log_perc_c = slope_log_perc_c * theoretical_quantiles_log_perc_c + intercept_log_perc_c

# Perform Bartlett test for dispersion for log(Perc_C)
bartlett_statistic_log_perc_c, bartlett_p_value_log_perc_c = bartlett(sorted_residuals_log_perc_c, trendline_log_perc_c)

# Plot the Q-Q plot with trend line and dispersion test results for log(Perc_C)
plt.figure(figsize=(8, 6))
plt.scatter(theoretical_quantiles_log_perc_c, sorted_residuals_log_perc_c, color='blue', alpha=0.5, label='Residuals log(Perc_C)')
plt.plot(theoretical_quantiles_log_perc_c, trendline_log_perc_c, color='red', linestyle='--', label='Trend line log(Perc_C)')
plt.text(0.5, -3, f'Bartlett statistic: {bartlett_statistic_log_perc_c:.2f}\nBartlett p-value: {bartlett_p_value_log_perc_c:.4f}', fontsize=10, ha='center')
plt.title('Q-Q Plot of Residuals for Encelia_californica with log(Perc_C) with Trend Line and Dispersion Test')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Residuals log(Perc_C)')
plt.legend()
plt.grid(True)
plt.show()


# In[46]:


#MALFAS PLOTS


# In[55]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import bartlett

# Load datasets
leaf_cn = pd.read_csv("Leaf_CN_r.csv")
sla = pd.read_csv("SLA.csv")
herb_mass = pd.read_csv("Herb_mass.csv")

# Merge datasets based on the common column "Plant"
merged_data = pd.merge(leaf_cn, sla, on="Plant")
merged_data = pd.merge(merged_data, herb_mass, on="Plant")

# Filter data for the specific species 'Artemisia_californica'
species_data = merged_data[merged_data['Species'] == 'Malacothamnus_fasciculatus']

# Take the logarithm of 'herb_mass_mg' and 'SLA'
species_data['log_herb_mass_mg'] = np.log(species_data['Herbivore_mass_mg'])
species_data['log_SLA'] = np.log(species_data['SLA'])

# Handle missing values
species_data.dropna(inplace=True)

# Define and fit the mixed effects model for SLA with log-transformed herbivore mass and log-transformed SLA
model_sla = sm.MixedLM.from_formula("log_herb_mass_mg ~ log_SLA", groups=species_data["Plot_x"] + species_data["Plant"], data=species_data)
result_sla = model_sla.fit()

# Calculate the residuals for SLA
residuals_sla = result_sla.resid

# Calculate the theoretical quantiles for SLA
sorted_residuals_sla = np.sort(residuals_sla)
n_sla = len(sorted_residuals_sla)
theoretical_quantiles_sla = sm.distributions.ECDF(sorted_residuals_sla)(sorted_residuals_sla)

# Fit a trend line for SLA
slope_sla, intercept_sla = np.polyfit(theoretical_quantiles_sla, sorted_residuals_sla, 1)
trendline_sla = slope_sla * theoretical_quantiles_sla + intercept_sla

# Perform Bartlett test for dispersion for SLA
bartlett_statistic_sla, bartlett_p_value_sla = bartlett(sorted_residuals_sla, trendline_sla)

# Plot the Q-Q plot with trend line and dispersion test results for SLA
plt.figure(figsize=(8, 6))
plt.scatter(theoretical_quantiles_sla, sorted_residuals_sla, color='blue', alpha=0.5, label='Residuals SLA')
plt.plot(theoretical_quantiles_sla, trendline_sla, color='red', linestyle='--', label='Trend line SLA')
plt.text(0.5, -3, f'Bartlett statistic: {bartlett_statistic_sla:.2f}\nBartlett p-value: {bartlett_p_value_sla:.4f}', fontsize=10, ha='center')
plt.title('Q-Q Plot of Residuals for Malacothamnus_fasciculatus with SLA (Log Transformed Herbivore Mass and SLA) with Trend Line and Dispersion Test')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Residuals SLA')
plt.legend()
plt.grid(True)
plt.show()


# In[56]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import bartlett

# Load datasets
leaf_cn = pd.read_csv("Leaf_CN_r.csv")
sla = pd.read_csv("SLA.csv")
herb_mass = pd.read_csv("Herb_mass.csv")

# Merge datasets based on the common column "Plant"
merged_data = pd.merge(leaf_cn, sla, on="Plant")
merged_data = pd.merge(merged_data, herb_mass, on="Plant")

# Filter data for the specific species 'Artemisia_californica'
species_data = merged_data[merged_data['Species'] == 'Malacothamnus_fasciculatus']

# Take the logarithm of herbivore_mass_mg
species_data['log_herb_mass_mg'] = np.log(species_data['Herbivore_mass_mg'])

# Handle missing values
species_data.dropna(inplace=True)

# Define and fit the mixed effects model for Perc_N with log-transformed herbivore mass
model_perc_n = sm.MixedLM.from_formula("log_herb_mass_mg ~ Perc_N", groups=species_data["Plot_x"] + species_data["Plant"], data=species_data)
result_perc_n = model_perc_n.fit()

# Calculate the residuals for Perc_N
residuals_perc_n = result_perc_n.resid

# Calculate the theoretical quantiles for Perc_N
sorted_residuals_perc_n = np.sort(residuals_perc_n)
n_perc_n = len(sorted_residuals_perc_n)
theoretical_quantiles_perc_n = sm.distributions.ECDF(sorted_residuals_perc_n)(sorted_residuals_perc_n)

# Fit a trend line for Perc_N
slope_perc_n, intercept_perc_n = np.polyfit(theoretical_quantiles_perc_n, sorted_residuals_perc_n, 1)
trendline_perc_n = slope_perc_n * theoretical_quantiles_perc_n + intercept_perc_n

# Perform Bartlett test for dispersion for Perc_N
bartlett_statistic_perc_n, bartlett_p_value_perc_n = bartlett(sorted_residuals_perc_n, trendline_perc_n)

# Plot the Q-Q plot with trend line and dispersion test results for Perc_N
plt.figure(figsize=(8, 6))
plt.scatter(theoretical_quantiles_perc_n, sorted_residuals_perc_n, color='blue', alpha=0.5, label='Residuals Perc_N')
plt.plot(theoretical_quantiles_perc_n, trendline_perc_n, color='red', linestyle='--', label='Trend line Perc_N')
plt.text(0.5, -3, f'Bartlett statistic: {bartlett_statistic_perc_n:.2f}\nBartlett p-value: {bartlett_p_value_perc_n:.4f}', fontsize=10, ha='center')
plt.title('Q-Q Plot of Residuals for Malacothamnus_fasciculatus with Perc_N (Log Transformed Herbivore Mass) with Trend Line and Dispersion Test')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Residuals Perc_N')
plt.legend()
plt.grid(True)
plt.show()

merged_data = pd.merge(leaf_cn, sla, on="Plant")
merged_data = pd.merge(merged_data, herb_mass, on="Plant")

# Filter data for the specific species 'Artemisia_californica'
species_data = merged_data[merged_data['Species'] == 'Malacothamnus_fasciculatus']

# Take the logarithm of herbivore_mass_mg
species_data['log_herb_mass_mg'] = np.log(species_data['Herbivore_mass_mg'])

# Handle missing values
species_data.dropna(inplace=True)

# Define and fit the mixed effects model for Perc_C with quadratic term
model_perc_c_quad = sm.MixedLM.from_formula("log_herb_mass_mg ~ Perc_C + np.power(Perc_C, 2)", groups=species_data["Plot_x"] + species_data["Plant"], data=species_data)
result_perc_c_quad = model_perc_c_quad.fit()

# Calculate the residuals for Perc_C
residuals_perc_c_quad = result_perc_c_quad.resid

# Calculate the theoretical quantiles for Perc_C
sorted_residuals_perc_c_quad = np.sort(residuals_perc_c_quad)
n_perc_c_quad = len(sorted_residuals_perc_c_quad)
theoretical_quantiles_perc_c_quad = sm.distributions.ECDF(sorted_residuals_perc_c_quad)(sorted_residuals_perc_c_quad)

# Fit a trend line for Perc_C
slope_perc_c_quad, intercept_perc_c_quad = np.polyfit(theoretical_quantiles_perc_c_quad, sorted_residuals_perc_c_quad, 1)
trendline_perc_c_quad = slope_perc_c_quad * theoretical_quantiles_perc_c_quad + intercept_perc_c_quad

# Perform Bartlett test for dispersion for Perc_C
bartlett_statistic_perc_c_quad, bartlett_p_value_perc_c_quad = bartlett(sorted_residuals_perc_c_quad, trendline_perc_c_quad)

# Plot the Q-Q plot with trend line and dispersion test results for Perc_C
plt.figure(figsize=(8, 6))
plt.scatter(theoretical_quantiles_perc_c_quad, sorted_residuals_perc_c_quad, color='blue', alpha=0.5, label='Residuals Perc_C (Quadratic)')
plt.plot(theoretical_quantiles_perc_c_quad, trendline_perc_c_quad, color='red', linestyle='--', label='Trend line Perc_C (Quadratic)')
plt.text(0.5, -3, f'Bartlett statistic: {bartlett_statistic_perc_c_quad:.2f}\nBartlett p-value: {bartlett_p_value_perc_c_quad:.4f}', fontsize=10, ha='center')
plt.title('Q-Q Plot of Residuals for Malacothamnus_fasciculatus with Perc_C (Quadratic) (Log Transformed Herbivore Mass) with Trend Line and Dispersion Test')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Residuals Perc_C (Quadratic)')
plt.legend()
plt.grid(True)
plt.show()


# In[57]:


#SALAPI PLOTS


# In[58]:


# Load datasets
leaf_cn = pd.read_csv("Leaf_CN_r.csv")
sla = pd.read_csv("SLA.csv")
herb_mass = pd.read_csv("Herb_mass.csv")

# Merge datasets based on the common column "Plant"
merged_data = pd.merge(leaf_cn, sla, on="Plant")
merged_data = pd.merge(merged_data, herb_mass, on="Plant")

# Filter data for the specific species 'Artemisia_californica'
species_data = merged_data[merged_data['Species'] == 'Salvia_apiana']

# Generate hypothetical count data for herbivores (e.g., herbivore sightings)
np.random.seed(0)
mean_herbivore_count = 5  # Adjust as needed
species_data['herbivore_count'] = np.random.poisson(mean_herbivore_count, size=len(species_data))

# Handle missing values
species_data.dropna(inplace=True)

# Define and fit the mixed effects model
model = smf.mixedlm("herbivore_count ~ Herbivore_mass_mg", species_data, groups=species_data["Plot_x"] + species_data["Plant"])
result = model.fit()

# Calculate the residuals
residuals = result.resid

# Calculate the theoretical quantiles
sorted_residuals = np.sort(residuals)
n = len(sorted_residuals)
theoretical_quantiles = sm.distributions.ECDF(sorted_residuals)(sorted_residuals)

# Fit a trend line
slope, intercept = np.polyfit(theoretical_quantiles, sorted_residuals, 1)
trendline = slope * theoretical_quantiles + intercept

# Perform Bartlett test for dispersion
bartlett_statistic, bartlett_p_value = bartlett(sorted_residuals, trendline)

# Plot the Q-Q plot with trend line and dispersion test results
plt.figure(figsize=(8, 6))
plt.scatter(theoretical_quantiles, sorted_residuals, color='blue', alpha=0.5, label='Residuals')
plt.plot(theoretical_quantiles, trendline, color='red', linestyle='--', label='Trend line')
plt.text(0.5, -3, f'Bartlett statistic: {bartlett_statistic:.2f}\nBartlett p-value: {bartlett_p_value:.4f}', fontsize=10, ha='center')
plt.title('Q-Q Plot of Residuals for Salvia_apiana with Trend Line and Dispersion Test')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Residuals')
plt.legend()
plt.grid(True)
plt.show()


# In[62]:


leaf_cn = pd.read_csv("Leaf_CN_r.csv")
sla = pd.read_csv("SLA.csv")
herb_mass = pd.read_csv("Herb_mass.csv")

# Merge datasets based on the common column "Plant"
merged_data = pd.merge(leaf_cn, sla, on="Plant")
merged_data = pd.merge(merged_data, herb_mass, on="Plant")

# Filter data for the specific species 'Artemisia_californica'
species_data = merged_data[merged_data['Species'] == 'Salvia_apiana']

# Generate hypothetical count data for herbivores (e.g., herbivore sightings)
np.random.seed(0)
mean_herbivore_count = 5  # Adjust as needed
species_data['herbivore_count'] = np.random.poisson(mean_herbivore_count, size=len(species_data))

# Handle missing values
species_data.dropna(inplace=True)

# Transform 'Perc_C' to its logarithm
species_data['log_Perc_N'] = np.log(species_data['Perc_N'])

# Define and fit the mixed effects model for log(Perc_C)
model_log_perc_n = smf.mixedlm("herbivore_count ~ log_Perc_N", species_data, groups=species_data["Plot_x"] + species_data["Plant"])
result_log_perc_n = model_log_perc_n.fit()

# Calculate the residuals for log(Perc_C)
residuals_log_perc_n = result_log_perc_n.resid

# Calculate the theoretical quantiles for log(Perc_C)
sorted_residuals_log_perc_n = np.sort(residuals_log_perc_n)
n_log_perc_c = len(sorted_residuals_log_perc_n)
theoretical_quantiles_log_perc_n = sm.distributions.ECDF(sorted_residuals_log_perc_n)(sorted_residuals_log_perc_n)

# Fit a trend line for log(Perc_C)
slope_log_perc_n, intercept_log_perc_n = np.polyfit(theoretical_quantiles_log_perc_n, sorted_residuals_log_perc_n, 1)
trendline_log_perc_n = slope_log_perc_n * theoretical_quantiles_log_perc_n + intercept_log_perc_n

# Perform Bartlett test for dispersion for log(Perc_C)
bartlett_statistic_log_perc_n, bartlett_p_value_log_perc_n = bartlett(sorted_residuals_log_perc_n, trendline_log_perc_n)

# Plot the Q-Q plot with trend line and dispersion test results for log(Perc_C)
plt.figure(figsize=(8, 6))
plt.scatter(theoretical_quantiles_log_perc_n, sorted_residuals_log_perc_n, color='blue', alpha=0.5, label='Residuals log(Perc_N)')
plt.plot(theoretical_quantiles_log_perc_n, trendline_log_perc_n, color='red', linestyle='--', label='Trend line log(Perc_N)')
plt.text(0.5, -3, f'Bartlett statistic: {bartlett_statistic_log_perc_n:.2f}\nBartlett p-value: {bartlett_p_value_log_perc_n:.4f}', fontsize=10, ha='center')
plt.title('Q-Q Plot of Residuals for Salvia_apiana with log(Perc_N) with Trend Line and Dispersion Test')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Residuals log(Perc_N)')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import poisson, bartlett

# Load datasets
leaf_cn = pd.read_csv("Leaf_CN_r.csv")
sla = pd.read_csv("SLA.csv")
herb_mass = pd.read_csv("Herb_mass.csv")

# Merge datasets based on the common column "Plant"
merged_data = pd.merge(leaf_cn, sla, on="Plant")
merged_data = pd.merge(merged_data, herb_mass, on="Plant")

# Filter data for the specific species 'Artemisia_californica'
species_data = merged_data[merged_data['Species'] == 'Salvia_apiana']

# Generate hypothetical count data for herbivores (e.g., herbivore sightings)
np.random.seed(0)
mean_herbivore_count = 5  # Adjust as needed
species_data['herbivore_count'] = np.random.poisson(mean_herbivore_count, size=len(species_data))

# Handle missing values
species_data.dropna(inplace=True)

# Transform 'Perc_C' to its logarithm
species_data['log_Perc_C'] = np.log(species_data['Perc_C'])

# Define and fit the mixed effects model for log(Perc_C)
model_log_perc_c = smf.mixedlm("herbivore_count ~ log_Perc_C", species_data, groups=species_data["Plot_x"] + species_data["Plant"])
result_log_perc_c = model_log_perc_c.fit()

# Calculate the residuals for log(Perc_C)
residuals_log_perc_c = result_log_perc_c.resid

# Calculate the theoretical quantiles for log(Perc_C)
sorted_residuals_log_perc_c = np.sort(residuals_log_perc_c)
n_log_perc_c = len(sorted_residuals_log_perc_c)
theoretical_quantiles_log_perc_c = sm.distributions.ECDF(sorted_residuals_log_perc_c)(sorted_residuals_log_perc_c)

# Fit a trend line for log(Perc_C)
slope_log_perc_c, intercept_log_perc_c = np.polyfit(theoretical_quantiles_log_perc_c, sorted_residuals_log_perc_c, 1)
trendline_log_perc_c = slope_log_perc_c * theoretical_quantiles_log_perc_c + intercept_log_perc_c

# Perform Bartlett test for dispersion for log(Perc_C)
bartlett_statistic_log_perc_c, bartlett_p_value_log_perc_c = bartlett(sorted_residuals_log_perc_c, trendline_log_perc_c)

# Plot the Q-Q plot with trend line and dispersion test results for log(Perc_C)
plt.figure(figsize=(8, 6))
plt.scatter(theoretical_quantiles_log_perc_c, sorted_residuals_log_perc_c, color='blue', alpha=0.5, label='Residuals log(Perc_C)')
plt.plot(theoretical_quantiles_log_perc_c, trendline_log_perc_c, color='red', linestyle='--', label='Trend line log(Perc_C)')
plt.text(0.5, -3, f'Bartlett statistic: {bartlett_statistic_log_perc_c:.2f}\nBartlett p-value: {bartlett_p_value_log_perc_c:.4f}', fontsize=10, ha='center')
plt.title('Q-Q Plot of Residuals for Salvia_apiana with log(Perc_C) with Trend Line and Dispersion Test')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Residuals log(Perc_C)')
plt.legend()
plt.grid(True)
plt.show()


# In[63]:


#SALMEL PLOTS


# In[66]:


leaf_cn = pd.read_csv("Leaf_CN_r.csv")
sla = pd.read_csv("SLA.csv")
herb_mass = pd.read_csv("Herb_mass.csv")

# Merge datasets based on the common column "Plant"
merged_data = pd.merge(leaf_cn, sla, on="Plant")
merged_data = pd.merge(merged_data, herb_mass, on="Plant")

# Filter data for the specific species 'Artemisia_californica'
species_data = merged_data[merged_data['Species'] == 'Salvia_mellifera']

# Generate hypothetical count data for herbivores (e.g., herbivore sightings)
np.random.seed(0)
mean_herbivore_count = 5  # Adjust as needed
species_data['herbivore_count'] = np.random.poisson(mean_herbivore_count, size=len(species_data))

# Handle missing values
species_data.dropna(inplace=True)

# Define and fit the mixed effects model
model = smf.mixedlm("herbivore_count ~ Herbivore_mass_mg", species_data, groups=species_data["Plot_x"] + species_data["Plant"])
result = model.fit()

# Calculate the residuals
residuals = result.resid

# Calculate the theoretical quantiles
sorted_residuals = np.sort(residuals)
n = len(sorted_residuals)
theoretical_quantiles = sm.distributions.ECDF(sorted_residuals)(sorted_residuals)

# Fit a trend line
slope, intercept = np.polyfit(theoretical_quantiles, sorted_residuals, 1)
trendline = slope * theoretical_quantiles + intercept

# Perform Bartlett test for dispersion
bartlett_statistic, bartlett_p_value = bartlett(sorted_residuals, trendline)

# Plot the Q-Q plot with trend line and dispersion test results
plt.figure(figsize=(8, 6))
plt.scatter(theoretical_quantiles, sorted_residuals, color='blue', alpha=0.5, label='Residuals')
plt.plot(theoretical_quantiles, trendline, color='red', linestyle='--', label='Trend line')
plt.text(0.5, -3, f'Bartlett statistic: {bartlett_statistic:.2f}\nBartlett p-value: {bartlett_p_value:.4f}', fontsize=10, ha='center')
plt.title('Q-Q Plot of Residuals for Salvia_mellifera with Trend Line and Dispersion Test')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Residuals')
plt.legend()
plt.grid(True)
plt.show()


# In[67]:


leaf_cn = pd.read_csv("Leaf_CN_r.csv")
sla = pd.read_csv("SLA.csv")
herb_mass = pd.read_csv("Herb_mass.csv")

# Merge datasets based on the common column "Plant"
merged_data = pd.merge(leaf_cn, sla, on="Plant")
merged_data = pd.merge(merged_data, herb_mass, on="Plant")

# Filter data for the specific species 'Artemisia_californica'
species_data = merged_data[merged_data['Species'] == 'Salvia_mellifera']

# Generate hypothetical count data for herbivores (e.g., herbivore sightings)
np.random.seed(0)
mean_herbivore_count = 5  # Adjust as needed
species_data['herbivore_count'] = np.random.poisson(mean_herbivore_count, size=len(species_data))

# Handle missing values
species_data.dropna(inplace=True)

# Transform 'Perc_C' to its logarithm
species_data['log_Perc_N'] = np.log(species_data['Perc_N'])

# Define and fit the mixed effects model for log(Perc_C)
model_log_perc_n = smf.mixedlm("herbivore_count ~ log_Perc_N", species_data, groups=species_data["Plot_x"] + species_data["Plant"])
result_log_perc_n = model_log_perc_n.fit()

# Calculate the residuals for log(Perc_C)
residuals_log_perc_n = result_log_perc_n.resid

# Calculate the theoretical quantiles for log(Perc_C)
sorted_residuals_log_perc_n = np.sort(residuals_log_perc_n)
n_log_perc_c = len(sorted_residuals_log_perc_n)
theoretical_quantiles_log_perc_n = sm.distributions.ECDF(sorted_residuals_log_perc_n)(sorted_residuals_log_perc_n)

# Fit a trend line for log(Perc_C)
slope_log_perc_n, intercept_log_perc_n = np.polyfit(theoretical_quantiles_log_perc_n, sorted_residuals_log_perc_n, 1)
trendline_log_perc_n = slope_log_perc_n * theoretical_quantiles_log_perc_n + intercept_log_perc_n

# Perform Bartlett test for dispersion for log(Perc_C)
bartlett_statistic_log_perc_n, bartlett_p_value_log_perc_n = bartlett(sorted_residuals_log_perc_n, trendline_log_perc_n)

# Plot the Q-Q plot with trend line and dispersion test results for log(Perc_C)
plt.figure(figsize=(8, 6))
plt.scatter(theoretical_quantiles_log_perc_n, sorted_residuals_log_perc_n, color='blue', alpha=0.5, label='Residuals log(Perc_N)')
plt.plot(theoretical_quantiles_log_perc_n, trendline_log_perc_n, color='red', linestyle='--', label='Trend line log(Perc_N)')
plt.text(0.5, -3, f'Bartlett statistic: {bartlett_statistic_log_perc_n:.2f}\nBartlett p-value: {bartlett_p_value_log_perc_n:.4f}', fontsize=10, ha='center')
plt.title('Q-Q Plot of Residuals for Salvia_mellifera with log(Perc_N) with Trend Line and Dispersion Test')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Residuals log(Perc_N)')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy.stats import poisson, bartlett

# Load datasets
leaf_cn = pd.read_csv("Leaf_CN_r.csv")
sla = pd.read_csv("SLA.csv")
herb_mass = pd.read_csv("Herb_mass.csv")

# Merge datasets based on the common column "Plant"
merged_data = pd.merge(leaf_cn, sla, on="Plant")
merged_data = pd.merge(merged_data, herb_mass, on="Plant")

# Filter data for the specific species 'Artemisia_californica'
species_data = merged_data[merged_data['Species'] == 'Salvia_mellifera']

# Generate hypothetical count data for herbivores (e.g., herbivore sightings)
np.random.seed(0)
mean_herbivore_count = 5  # Adjust as needed
species_data['herbivore_count'] = np.random.poisson(mean_herbivore_count, size=len(species_data))

# Handle missing values
species_data.dropna(inplace=True)

# Transform 'Perc_C' to its logarithm
species_data['log_Perc_C'] = np.log(species_data['Perc_C'])

# Define and fit the mixed effects model for log(Perc_C)
model_log_perc_c = smf.mixedlm("herbivore_count ~ log_Perc_C", species_data, groups=species_data["Plot_x"] + species_data["Plant"])
result_log_perc_c = model_log_perc_c.fit()

# Calculate the residuals for log(Perc_C)
residuals_log_perc_c = result_log_perc_c.resid

# Calculate the theoretical quantiles for log(Perc_C)
sorted_residuals_log_perc_c = np.sort(residuals_log_perc_c)
n_log_perc_c = len(sorted_residuals_log_perc_c)
theoretical_quantiles_log_perc_c = sm.distributions.ECDF(sorted_residuals_log_perc_c)(sorted_residuals_log_perc_c)

# Fit a trend line for log(Perc_C)
slope_log_perc_c, intercept_log_perc_c = np.polyfit(theoretical_quantiles_log_perc_c, sorted_residuals_log_perc_c, 1)
trendline_log_perc_c = slope_log_perc_c * theoretical_quantiles_log_perc_c + intercept_log_perc_c

# Perform Bartlett test for dispersion for log(Perc_C)
bartlett_statistic_log_perc_c, bartlett_p_value_log_perc_c = bartlett(sorted_residuals_log_perc_c, trendline_log_perc_c)

# Plot the Q-Q plot with trend line and dispersion test results for log(Perc_C)
plt.figure(figsize=(8, 6))
plt.scatter(theoretical_quantiles_log_perc_c, sorted_residuals_log_perc_c, color='blue', alpha=0.5, label='Residuals log(Perc_C)')
plt.plot(theoretical_quantiles_log_perc_c, trendline_log_perc_c, color='red', linestyle='--', label='Trend line log(Perc_C)')
plt.text(0.5, -3, f'Bartlett statistic: {bartlett_statistic_log_perc_c:.2f}\nBartlett p-value: {bartlett_p_value_log_perc_c:.4f}', fontsize=10, ha='center')
plt.title('Q-Q Plot of Residuals for Salvia_mellifera with log(Perc_C) with Trend Line and Dispersion Test')
plt.xlabel('Theoretical Quantiles')
plt.ylabel('Residuals log(Perc_C)')
plt.legend()
plt.grid(True)
plt.show()


# In[68]:


#ANOVA Tables


# In[79]:


import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Load datasets
leaf_cn = pd.read_csv("Leaf_CN_r.csv")
sla = pd.read_csv("SLA.csv")
herb_mass = pd.read_csv("Herb_mass.csv")

# Merge datasets based on the common column "Plant"
merged_data = pd.merge(leaf_cn, sla, on="Plant")
merged_data = pd.merge(merged_data, herb_mass, on="Plant")

# Choose a species
species = 'Salvia_mellifera'

# Filter data for the chosen species
species_data = merged_data[merged_data['Species'] == species]

# Prepare the formula for the null model
formula_null = 'Herbivore_mass_mg ~ 1'

# Prepare the formula for the alternative model
formula_alt = 'Herbivore_mass_mg ~ SLA + Perc_N + np.log(Perc_C)'

# Fit the null model
null_model = smf.mixedlm(formula_null, species_data, groups=species_data["Plot_x"] + species_data["Plant"])
null_result = null_model.fit()

# Fit the alternative model
alt_model = smf.mixedlm(formula_alt, species_data, groups=species_data["Plot_x"] + species_data["Plant"])
alt_result = alt_model.fit()

# Perform likelihood ratio test
lr_test = alt_result.compare_lr_test(null_result)
chi_squared = lr_test[0]
p_value = lr_test[1]

# Print results
print(f"Chi-squared value: {chi_squared:.4f} (p-value < {p_value:.4f})")


# In[41]:


merged_data.head(100)


# In[ ]:




