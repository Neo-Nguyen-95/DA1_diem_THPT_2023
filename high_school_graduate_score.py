"""Objective: Analyze high school graduate score of students in Hanoi 2023
"""
#%% LIBRARY
import pandas as pd
pd.options.display.max_columns = None

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
pio.renderers.default="browser"
#%% IMPORT & CLEAN
df = pd.read_csv("diem_thi_thpt_2023.csv")
df = df[df['ma_ngoai_ngu'] == "N1"]  # only English

df["region"] = df["sbd"].map(lambda x: int(x//1e6))  # first 2 numbers is region ID
df.drop(columns=['ma_ngoai_ngu', 'sbd'], inplace=True)  # done with sbd & ma nn

# encode region with economic development
# 2 = developed, 1 = average, 0 = poor
region = pd.read_csv("region_map.csv")
region = region.set_index("region")

# update region economic
df['economic'] = df['region'].map(lambda x: region.loc[x]['economic'])

# Group 1: Nature Science Score
df_science = df.drop(columns=['lich_su', 'dia_li', 'gdcd'])
df_science.dropna(inplace=True)

# Group 2: Social Science Score
df_temp = df.drop(columns=['vat_li', 'hoa_hoc', 'sinh_hoc'])
df_social = df_temp[df_temp['gdcd'].notna()]
df_social.dropna(inplace=True)

# Group 3: GDTX score
df_gdtx = df_temp[df_temp['gdcd'].isna()]
df_gdtx.drop(columns='gdcd', inplace=True)
df_gdtx.dropna(inplace=True)

# Summary
total_num = len(df_science) + len(df_social) + len(df_gdtx)
print(f"{len(df_science)} student on Science major, \
make up {len(df_science) / total_num * 100:.2f}%")
print(f"{len(df_social)} student on Social major, \
make up {len(df_social) / total_num * 100:.2f}%")
print(f"{len(df_gdtx)} student on GDTX major, \
make up {len(df_gdtx) / total_num * 100:.2f}%")

df_update = pd.concat([df_science, df_social])

#%% FIRST EDA
# Imbalance check on geography region
temp = df_update['region'].value_counts(normalize=True);
temp = pd.DataFrame(temp).join(region)
top4 = temp.reset_index().iloc[:5, 1:3]
others = temp.reset_index().iloc[5:, 1:3]
others = pd.DataFrame({
    'proportion' : [others['proportion'].sum()],
    'name': ['KHU VỰC KHÁC']
    })
temp = pd.concat([top4, others], axis="rows")
plt.figure(figsize=(10, 10), dpi=600)
plt.pie(temp.proportion, labels=temp.name, autopct='%1.0f%%',
        radius = 0.75,rotatelabels=True,
        colors=['aquamarine', 'turquoise', 'gold', 'coral', 'pink', 'lavender']);

# Imbalance check on economic region
temp = df_update['economic'].value_counts(normalize=True);
temp = pd.DataFrame({
    "economic": temp,
    "name": ['Vùng 1', 'Vùng 3', 'Vùng 2', 'Vùng 0']
    })
plt.figure(figsize=(5, 5), dpi=600)
plt.pie(temp.economic, labels=temp.name, autopct='%1.0f%%',
        radius = 0.75,
        colors=['aquamarine', 'orange', 'turquoise', 'salmon']);

# Average score check
temp = df_update.drop(columns=['economic','region'])
plt.figure(figsize=(8, 6), dpi=600)
sns.boxplot(data=temp,palette='deep', width=.5, orient='h',
            flierprops={'markerfacecolor':'1',
                        'markersize': 3
                         })
#%% EDA SUBJECT SCORE DISTRIBUTION BY REGION
# Spread of data: Math, Literature, English
plt.figure(figsize=(5, 5), dpi=600)
sns.kdeplot(data=df_update, x='toan', hue='economic', palette='deep')
plt.title("Distribution of Math scores")
plt.xlabel("Math score")
plt.ylabel("Density")

plt.figure(figsize=(5, 5), dpi=600)
sns.kdeplot(data=df_update, x='ngu_van', hue='economic', palette='deep')
plt.title("Distribution of Literature scores")
plt.xlabel("Literature score")
plt.ylabel("Density")

plt.figure(figsize=(5, 5), dpi=600)
sns.kdeplot(data=df_update, x='ngoai_ngu', hue='economic', palette='deep')
plt.title("Distribution of English scores")
plt.xlabel("English score")
plt.ylabel("Density")

plt.figure(figsize=(10, 5), dpi=600)
temp = df_update[(df_update['economic']==3)]['region'].map(
    {1:'high', 2:'high', 4:'high', 44:'high', 3:'low',17:'low',55:'low'})
sns.kdeplot(data=df_update[df_update['economic']==3],
            x='ngoai_ngu', hue=temp, palette='deep')
plt.title("Distribution of English scores")
plt.xlabel("English score")
plt.ylabel("Density")

plt.figure(figsize=(10, 5), dpi=600)
temp = df_update[(df_update['economic']==3) & (df_update['region']<3)]['region'].map(
    {1:'HÀ NỘI', 2:'HỒ CHÍ MINH'})
sns.kdeplot(data=df_update[(df_update['economic']==3) & (df_update['region']<3)],
            x='ngoai_ngu', hue=temp, palette='deep')
plt.title("Distribution of English scores")
plt.xlabel("English score")
plt.ylabel("Density")

# Spread of data: Physics, Chemistry, Biology
plt.figure(figsize=(5, 5), dpi=600)
sns.kdeplot(data=df_science, x='vat_li', hue='economic', palette='deep')
plt.title("Distribution of Physics scores")
plt.xlabel("Physics score")
plt.ylabel("Density")

plt.figure(figsize=(5, 5), dpi=600)
sns.kdeplot(data=df_science, x='hoa_hoc', hue='economic', palette='deep')
plt.title("Distribution of Chemistry scores")
plt.xlabel("Chemitry score")
plt.ylabel("Density")

plt.figure(figsize=(5, 5), dpi=600)
sns.kdeplot(data=df_science, x='sinh_hoc', hue='economic', palette='deep')
plt.title("Distribution of Biology scores")
plt.xlabel("Biology score")
plt.ylabel("Density")

# Spread of data: Geography, History, Humanity
plt.figure(figsize=(5, 5), dpi=600)
sns.kdeplot(data=df_social, x='dia_li', hue='economic', palette='deep')
plt.title("Distribution of Geography scores")
plt.xlabel("Score")
plt.ylabel("Density")

plt.figure(figsize=(5, 5), dpi=600)
sns.kdeplot(data=df_social, x='lich_su', hue='economic', palette='deep')
plt.title("Distribution of History scores")
plt.xlabel("Score")
plt.ylabel("Density")

plt.figure(figsize=(5, 5), dpi=600)
sns.kdeplot(data=df_social, x='gdcd', hue='economic', palette='deep')
plt.title("Distribution of Humanity scores")
plt.xlabel("Score")
plt.ylabel("Density")

#%% EDA ENGLISH MAP
region_mean_score = df_update.groupby("region")["ngoai_ngu"].mean()
top10 = region_mean_score.sort_values(ascending=False).iloc[0:10]
top10 = pd.DataFrame(top10).join(region).drop(columns=['lat', 'lon'])
print(top10['economic'].value_counts(normalize=True))

region_mean_score = pd.DataFrame(region_mean_score)

eng_region = region_mean_score.join(region)

fig = px.scatter_mapbox(
    eng_region,  # Our DataFrame
    lat='lat', lon='lon', 
    color='ngoai_ngu', color_continuous_scale="OrRd",
    size=(eng_region["ngoai_ngu"]*20), size_max=10,
    center={"lat": 16.740751, "lon": 107.017227} , 
    width=700, height=900,
    hover_data="ngoai_ngu", hover_name="name",
    zoom=4.5,
    title="English mean score by region"
)

# Add mapbox_style to figure layout
fig.update_layout(mapbox_style="carto-positron")

# Show figure
fig.show()

#%% EDA CORRELATION
# Correlation between subject and region
economic_corr = df_update.corr()
economic_corr = economic_corr.iloc[[0, 1, 2, 3, 4, 5, 8 ,9, 10],7]
plt.figure(figsize=(8, 2), dpi=600)
sns.heatmap(data=pd.DataFrame(economic_corr).T, annot=True, fmt=".2f",
            cmap='Greens')
plt.title("Correlation between subject scores and economic")
plt.yticks(rotation=0)


# Correlation between subject
subject_corr = df_update.iloc[:,[0, 1, 2, 3, 4, 5, 8 ,9, 10]].corr()
plt.figure(figsize=(10, 5), dpi=600)
sns.heatmap(data=subject_corr, annot=True, fmt=".1f")
plt.title("Correlation between subject scores")

# Correlation between science group
subject_corr = df_science.iloc[:,[0, 1, 2, 3, 4, 5]].corr()
plt.figure(figsize=(10, 5), dpi=600)
sns.heatmap(data=subject_corr, annot=True, fmt=".1f")
plt.title("Correlation between subject scores")

# Correlation between social group
subject_corr = df_social.iloc[:,[0, 1, 2, 3, 4, 5]].corr()
plt.figure(figsize=(10, 5), dpi=600)
sns.heatmap(data=subject_corr, annot=True, fmt=".1f")
plt.title("Correlation between subject scores")
