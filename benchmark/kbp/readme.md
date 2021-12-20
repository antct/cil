## Wiki-KBP

Paper: http://xiaoling.github.io/pubs/ling-aaai12.pdf

Unprocessed Data: [Google Drive](https://drive.google.com/drive/folders/0B--ZKWD8ahE4RjFLUkVQTm93WVU?usp=sharing)

Data Format Example:

```
kbp_rel2id.json
{"no_relation": 0, "per:title": 1, "org:top_members/employees": 2, "per:employee_of": 3, "org:alternate_names": 4, "org:country_of_headquarters": 5, "per:countries_of_residence": 6, "org:city_of_headquarters": 7, "per:cities_of_residence": 8, "per:age": 9, "per:stateorprovinces_of_residence": 10, "per:origin": 11, "org:subsidiaries": 12, "org:parents": 13, "per:spouse": 14, "org:stateorprovince_of_headquarters": 15, "per:children": 16, "per:other_family": 17, "per:alternate_names": 18, "org:members": 19, "per:siblings": 20, "per:schools_attended": 21, "per:parents": 22, "per:date_of_death": 23, "org:member_of": 24, "org:founded_by": 25, "org:website": 26, "per:cause_of_death": 27, "org:political/religious_affiliation": 28, "org:founded": 29, "per:city_of_death": 30, "org:shareholders": 31, "org:number_of_employees/members": 32, "per:date_of_birth": 33, "per:city_of_birth": 34, "per:charges": 35, "per:stateorprovince_of_death": 36, "per:religion": 37, "per:stateorprovince_of_birth": 38, "per:country_of_birth": 39, "org:dissolved": 40, "per:country_of_death": 41}

kbp_train.txt kbp_dev.txt kbp_test.txt
{"text": "An American Civil rights activist , Martin Luther King , Jr. was assassinated in 1968 by James Earl Ray in Memphis , Tennessee .", "relation": "no_relation", "h": {"id": "1-a", "name": "American Civil", "pos": [3, 17]}, "t": {"id": "1-b", "name": "James Earl Ray", "pos": [89, 103]}}
```