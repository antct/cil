## Google IISc Distant Supervision (GIDS)

Paper: https://arxiv.org/pdf/1804.06987.pdf

Unprocessed Data: [Google Drive](https://drive.google.com/open?id=1gTNAbv8My2QDmP-OHLFtJFlzPDoCG4aI)

Data Format Example:

```
gds_rel2id.json
{"NA": 0, "/people/deceased_person/place_of_death": 1, "/people/person/place_of_birth": 2, "/people/person/education./education/education/institution": 3, "/people/person/education./education/education/degree": 4}

gds_train.txt gds_dev.txt gds_test.txt
{"relation": "/people/deceased_person/place_of_death", "text": "War as appropriate. Private Alfred James_Smurthwaite Sample. 26614. 2nd Battalion Yorkshire Regiment. Son of Edward James Sample, of North_Ormesby , Yorks. Died 2 April 1917. Aged 29. Born Ormesby, Enlisted Middlesbrough. Buried BUCQUOY ROAD CEMETERY, FICHEUX. Not listed on the Middlesbrough War Memorial Private Frederick Scott. 46449. 4th Battalion Yorkshire Regiment. Son of William and Maria Scott, of 25, Aspinall St., Heywood, Lancs. Born at West Hartlepool. Died 27 May 1918. Aged 24.", "h": {"id": "/m/02qt0sv", "name": "James_Smurthwaite", "pos": [35, 52]}, "t": {"id": "/m/0fnhl9", "name": "North_Ormesby", "pos": [133, 146]}}
```