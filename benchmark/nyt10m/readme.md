## NYT10 Manual

Paper: https://pdfs.semanticscholar.org/db55/0f7af299157c67d7f1874bf784dca10ce4a9.pdf

Processed Data: [Google Drive](https://drive.google.com/drive/folders/0B--ZKWD8ahE4UktManVsY1REOUk?usp=sharing)

Data Format Example:

```
nyt10m_rel2id.json
{"no_relation": 0, "/location/location/contains": 1, "/people/person/place_of_birth": 2, "/people/deceased_person/place_of_death": 3, "/people/person/nationality": 4, "/people/person/place_lived": 5, "/business/person/company": 6, "/location/country/administrative_divisions": 7, "/location/administrative_division/country": 8, "/business/company/founders": 9, "/location/country/capital": 10, "/location/neighborhood/neighborhood_of": 11, "/sports/sports_team/location": 12, "/sports/sports_team_location/teams": 13, "/business/company/place_founded": 14, "/people/person/children": 15, "/people/person/religion": 16, "/people/ethnicity/geographic_distribution": 17, "/business/company/major_shareholders": 18, "/business/company_shareholder/major_shareholder_of": 19, "/business/company/advisors": 20, "/people/ethnicity/people": 21, "/people/person/ethnicity": 22, "/people/person/profession": 23, "/business/company/industry": 24}

nyt10m_train.txt nyt10m_dev.txt nyt10m_test.txt
{"text": "Kerry Packer , who became Australia 's richest man by turning a magazine and television inheritance worth millions into a diverse business worth billions , died yesterday in Sydney .", "relation": "/people/deceased_person/place_of_death", "h": {"id": "1-a", "name": "Kerry Packer", "pos": [0, 12]}, "t": {"id": "1-b", "name": "Sydney", "pos": [174, 180]}}
```