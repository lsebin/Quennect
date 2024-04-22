import mysql.connector

def fetch_db(id, state):
  host = "quennect.ckgyy9i5kf2y.us-east-1.rds.amazonaws.com"
  user = "quennect"
  password = "2-quennect-casss"
  database = "QUENNECT"
  port = 3306
  try:
          connection = mysql.connector.connect(
          host=host,
          user=user,
          password=password,
          database=database,
          port=port
          )
          
          cursor = connection.cursor()
          
          def execute_query(query, id, selectquery, state):
            cursor.execute(query, (id,))
            result = cursor.fetchone()
            if result is None or any(value is None for value in result):
              cursor.execute(selectquery + " FROM states WHERE STATE_ABBR = %s", (state,))
              result = cursor.fetchone()
            return result
          
          
                    
          '''
          coord_result = execute_query("SELECT latitude, longitude FROM usaCounties WHERE ID = %s", 
                                       id, "SELECT latitude, longitude", state)
          '''
          coord_result = execute_query("SELECT latitude, longitude FROM final WHERE ID = %s", 
                                       id, "SELECT latitude, longitude", state)
          #latitude = coord_result[0]
          #longitude = coord_result[1]
          latitude, longitude = coord_result if coord_result else (None, None)
          
          pop_result = execute_query("SELECT POP_SQMI FROM final WHERE ID = %s", id, "SELECT POP_SQMI", state)
          pop_density = pop_result[0] if pop_density else None
          
          '''
          voting_result = execute_query("SELECT votes_dem, votes_rep, votes_total, votes_per_sqkm, pct_dem_lead FROM precincts WHERE ID = %s", 
                                       id, "SELECT votes_dem, votes_rep, votes_total, votes_per_sqkm, pct_dem_lead", state)
          '''
          voting_result = execute_query("SELECT votes_dem, votes_rep, votes_total, votes_per_sqkm, pct_dem_lead FROM final WHERE ID = %s", 
                                       id, "SELECT votes_dem, votes_rep, votes_total, votes_per_sqkm, pct_dem_lead", state)
          '''
          votes_dem = voting_result[0]
          votes_rep = voting_result[1]
          votes_total = voting_result[2]
          votes_per_sqkm = voting_result[3]
          pct_dem_lead = voting_result[4]
          '''
          votes_dem, votes_rep, votes_total, votes_per_sqkm, pct_dem_lead = voting_result if voting_result else (None, None, None, None, None)
          
          '''
          ghi_result = execute_query("SELECT GHI FROM GHI WHERE ID = %s", 
                                      id, "SELECT GHI", state)
          '''
          ghi_result = execute_query("SELECT GHI FROM final WHERE ID = %s", 
                                      id, "SELECT GHI", state)
          #ghi = ghi_result[0]
          ghi = ghi_result[0] if ghi_result else None
          
          '''
          windSpeed_result = execute_query("SELECT windSpeed FROM windSpeed WHERE ID = %s", 
                                            id, "SELECT GHI", state)
          '''
          windSpeed_result = execute_query("SELECT windSpeed FROM final WHERE ID = %s", 
                                            id, "SELECT GHI", state)
          #windSpeed = windSpeed_result[0]
          windSpeed = windSpeed_result[0] if windSpeed_result else None
              
          cursor.close()
          connection.close()
                    
          return [latitude, longitude, pop_density, votes_dem, votes_rep, votes_total, votes_per_sqkm, pct_dem_lead, ghi, windSpeed]
  except Exception as e:
          print(f"Error fetching data from database: {e}")
          return None