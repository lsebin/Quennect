import json
import time
import pandas as pd
from flask import Flask, request, jsonify
from run_model import run_shap
from run_lime import run_lime, scale_user_input
from db import fetch_db
from apicall import get_analysis
import mysql.connector
#NOTE: don't forget to run 'pip install mysql-connector-python'

host = "quennect"
user = "quennect"
password = "2-quennect-casss"
database = "quennect.ckgyy9i5kf2y.us-east-1.rds.amazonaws.com"
port = 3306
 
app = Flask(__name__)

@app.route('/')
def home():
    return "home"

@app.route('/api/model', methods=['POST'])
def model():
    data = request.json
    utility = data['utility']
    region = data['region']
    size = data['size']
    energy = data['energy']
    state = data['state']
    county = data['county']
    #latitude = data['latitude']
    #longitude = data['longitude']
    year = data['year']
    
    print(', '.join(map(str, data)))
        
    id = county + ',' + state
    
    fetched = fetch_db(id, state)
    
    latitude, longitude, votes_dem, votes_rep, votes_total, votes_per_sqkm, pct_dem_lead, ghi, windSpeed = fetched
    
    from_user = [0 for _ in range(30)]
    
    #TODO: difference btwn 'year_entering_queue' & 'proposed_year'?
    #TODO: check if 2024 works. Consider hard-coding value (2023 will be safe)
    from_user[0] = year
    from_user[1] = year
    
    if region == 'CAISO':
        from_user[2] = 1
    elif region == 'MISO':
        from_user[3] = 1
    elif region == 'PJM':
        from_user[4] = 1
    elif region == 'Southeast (non-ISO)':
        from_user[5] = 1
    elif region == 'West (non-ISO)':
        from_user[6] = 1
        
    from_user[7] = size
    
    from_user[8] = latitude
    from_user[9] = longitude
    
    #TODO: fix population density
    from_user[10] = votes_per_sqkm * 0.386102
    
    from_user[11] = votes_dem
    from_user[12] = votes_rep
    from_user[13] = votes_total
    from_user[14] = votes_per_sqkm  # voting_density
    from_user[15] = pct_dem_lead
    
    #from_user[16] = solar_potential
    #from_user[17] = wind_potential
    
    '''
    ['year_entering_queue', 'proposed_year', 'region_CAISO', 
    'region_MISO', 'region_PJM', 'region_Southeast (non-ISO)', '
    region_West (non-ISO)', 'project_size_mw', 'project_latitude',
    'project_longitude', 'population_density', 'votes_dem', 
    'votes_rep', 'votes_total', 'voting_density',
    'pct_dem_lead', 'solar_potential', 'wind_potential', 
    'is_deregulated','has_100_clean_energy_goal','top_ten_renewable_generators', 
    'is_solar', 'is_storage', 'is_wind', 
    'is_bioenergy', 'is_wasteuse', 'is_cleanenergy', 
    'is_fossilfuels', 'is_hybrid', 'high_revenue_utility']
    '''
    
    #is_deregulated
    deregulated = ['OR', 'CA', 'TX', 'IL', 'MI', 'OH', 'VA', 'MD', 'DE', 'PA', 'NJ', 'NY', 'MA', 'CT', 'RI', 'NH', 'ME']
    
    for entry in deregulated:
        if energy == entry:
            from_user[18] = 1
            
    cleanenergy_goal = ['CA', 'CO', 'CT', 'DE', 'HI', 'IL', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'NE', 'NV', 'NJ', 'NM', 'NY', 'NC', 'OR', 'RI', 'VA', 'WA', 'WI']
    
    for entry in cleanenergy_goal:
        if energy == entry:
            from_user[19] = 1
            
    topten = ['TX', 'FL', 'PA', 'CA', 'IL', 'AL', 'OH', 'NC', 'GA', 'NY']
    
    for entry in topten:
        if energy == entry:
            from_user[20] = 1
    
    if energy == 'Solar':
        from_user[21] = 1

    storage = ['Battery', 'Hydro', 'Gravity Rail', 'Flywheel', 'Pumped Storage']
    
    for entry in storage:
        if energy == entry:
            from_user[22] = 1
    
    if (energy == 'Wind') or (energy == 'Offshore Wind'):
        from_user[23] = 1
    
    bioenergy = ['Biofuel', 'Biogas', 'Biomass', 'Wood']
    
    for entry in bioenergy:
        if energy == entry:
            from_user[24] = 1

    wasteuse = ['Landfill', 'Methane', 'Waste Heat']
    
    for entry in wasteuse:
        if energy == entry:
            from_user[25] = 1
            
    cleanenergy = ['Geothermal', 'Nuclear', 'Solar', 'Offshore Wind', 'Hydro', 'Wind']
            
    for entry in cleanenergy:
        if energy == entry:
            from_user[26] = 1

    fossil = ['Coal', 'Diesel', 'Gas', 'Oil', 'Steam']
    
    for entry in fossil:
        if energy == entry:
            from_user[27] = 1
    
    if region == 'Hybrid':
        from_user[28] = 1
        
    # high_revenue utility
    high_revenue = ['SOCO', 'Duke Energy Indiana, LLC', 
     'Duke_FL','Duke Energy Corporation','Duke Energy', 
     'Duke', 'PGE', 'AEP', 'DominionSC', 'Dominion SC', 'Dominion']
    
    for entry in high_revenue:
        if utility == entry:
            from_user[29] = 1
    
    
    
    X_train = pd.read_csv('X_train.csv')
    X_train = X_train.drop('Unnamed: 0', axis=1)
    scaled = scale_user_input(from_user)
    X_test = pd.Series((x for x in scaled[0]))
    
    top_5_features_rec = run_shap(X_test, X_train)
    # print(top_5_features_rec)
    
    predict, features_weight = run_lime(X_test, X_train)
    rec = None
    
    #TODO: connecting with openai api
    
    #print(predict)
    #print(features_weight)
    
    #if predict < 0.8:
    #    rec = top_5_features_rec
    
    rec = top_5_features_rec
    
    raw_dic = {feature_names[i]: from_user[i] for i in range(len(from_user))}
    
    fw_json = '\n'.join([f'"{key}": {value}' for key, value in features_weight.items()])
    rd_json = '\n'.join([f'"{key}": {value}' for key, value in raw_dic.items()])
    r_json = '\n'.join([f'"{key}": {value}' for key, value in rec.items()])
    
    data = {'Explanation': fw_json, 'Raw Data': rd_json, 'Recommendation': r_json}
    
    r_bracket = '{'
    l_bracket = '}'
    latter = '\n'.join([f'{key}:{r_bracket}\n{value}{l_bracket}' for key, value in data.items()])
    formatted_json = f'Prediction : {predict} \n{latter}'
    
    debug = ', '.join(map(str, from_user))
    
    txt = get_analysis(formatted_json)
    
    #return jsonify({'txt': txt, 'debug': debug})
    return jsonify({'txt': txt})
    
    #return jsonify({'recommedation': rec, 'features': features_weight})
    #return jsonify({'analysis': analysis, 'debug': debug})

if __name__ == "__main__":
    app.run(debug=True, port=5000)