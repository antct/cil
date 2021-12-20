## NYT10 Distant

Paper: http://iesl.cs.umass.edu/riedel/ecml/

Processed Data: [OpenNRE](https://github.com/thunlp/OpenNRE/blob/master/benchmark/download_nyt10.sh)

Data Format Example:

```
nyt10d_rel2id.json
{"/location/fr_region/capital": 2, "/location/cn_province/capital": 3, "/location/in_state/administrative_capital": 4, "/base/locations/countries/states_provinces_within": 5, "/business/company/founders": 6, "/people/person/place_of_birth": 8, "/people/deceased_person/place_of_death": 9, "/location/it_region/capital": 10, "/people/family/members": 11, "/people/profession/people_with_this_profession": 14, "/location/neighborhood/neighborhood_of": 1, "NA": 0, "/location/in_state/legislative_capital": 16, "/sports/sports_team/location": 17, "/people/person/religion": 18, "/location/in_state/judicial_capital": 19, "/business/company_advisor/companies_advised": 20, "/people/family/country": 21, "/time/event/locations": 22, "/business/company/place_founded": 23, "/location/administrative_division/country": 24, "/people/ethnicity/included_in_group": 25, "/location/br_state/capital": 15, "/location/mx_state/capital": 26, "/location/province/capital": 27, "/people/person/nationality": 28, "/business/person/company": 29, "/business/shopping_center_owner/shopping_centers_owned": 30, "/business/company/advisors": 31, "/business/shopping_center/owner": 32, "/location/country/languages_spoken": 7, "/people/deceased_person/place_of_burial": 34, "/location/us_county/county_seat": 13, "/people/ethnicity/geographic_distribution": 35, "/people/person/place_lived": 36, "/business/company/major_shareholders": 37, "/broadcast/producer/location": 38, "/location/us_state/capital": 12, "/broadcast/content/location": 39, "/business/business_location/parent_company": 40, "/location/jp_prefecture/capital": 41, "/film/film/featured_film_locations": 42, "/people/place_of_interment/interred_here": 43, "/location/de_state/capital": 44, "/people/person/profession": 45, "/business/company/locations": 46, "/location/country/capital": 47, "/location/location/contains": 48, "/people/person/ethnicity": 33, "/location/country/administrative_divisions": 49, "/people/person/children": 50, "/film/film_location/featured_in_films": 51, "/film/film_festival/location": 52}

nyt10d_train.txt nyt10d_dev.txt nyt10d_test.txt
{"text": "sen. charles e. schumer called on federal safety officials yesterday to reopen their investigation into the fatal crash of a passenger jet in belle harbor , queens , because equipment failure , not pilot error , might have been the cause .", "relation": "/location/location/contains", "h": {"id": "m.0ccvx", "name": "queens", "pos": [157, 163]}, "t": {"id": "m.05gf08", "name": "belle harbor", "pos": [142, 154]}}
```