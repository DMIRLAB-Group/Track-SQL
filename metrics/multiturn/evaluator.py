import os
import sys
from metrics.multiturn.evaluation import build_foreign_key_map_from_json, evaluate


class Evaluator:
    def __init__(self, table_path, db_dir):
        self.table_path = table_path
        self.db_dir = db_dir
        self.kmaps = build_foreign_key_map_from_json(self.table_path)

    def change_database(self, db_dir):
        self.db_dir = db_dir

    def accuracy(self, pred_filename, dataset, output_path, etype='all'):
        assert etype in ['match', 'exec', 'all']
        result = self.evaluate_with_official_interface(pred_filename, dataset, output_path, etype)
        if etype == 'match':
            return float(result['exact'])
        if etype == 'exec':
            return float(result['exec'])
        return (float(result['exact']), float(result['exec']))

    def evaluate_with_official_interface(self, pred_filename, dataset_filepath, etype='all'):
        import json
        golds = []
        with open(dataset_filepath, encoding="utf-8") as f:
            dataset = json.load(f)
            for id, samples in enumerate(dataset):
                interaction = samples["interaction"]
                db_id = samples["database_id"]
                tmp = []
                for turn_idx, sample in enumerate(interaction):
                    if sample['query'] == "SELECT T1.id, T1.name FROM battle EXCEPT SELECT T1.id, T1.name FROM battle AS T1 JOIN ship AS T2 ON T1.id  =  T2.lost_in_battle WHERE T2.location  =  'English Channel'":
                        sample['query'] = "SELECT id, name FROM battle EXCEPT SELECT T1.id, T1.name FROM battle AS T1 JOIN ship AS T2 ON T1.id  =  T2.lost_in_battle WHERE T2.location  =  'English Channel'"
                    elif sample['query'] == "SELECT countryname FROM countries WHERE countryid = 1 or countryid = 2 or countryid = 3 ) ":
                        sample['query'] = "SELECT countryname FROM countries WHERE countryid = 1 or countryid = 2 or countryid = 3"
                    elif sample['query'] == "SELECT T1.Name FROM country AS t1 WHERE  IndepYear  <  1930" and sample["utterance"] == "What are the names of those countries that were founded after 1930?":
                        sample['query'] = "SELECT T1.Name FROM country AS t1 WHERE  IndepYear  >  1930"
                    elif sample['query'] == "SELECT min(population) FROM country WHERE Continent  =  \"Asia\"" and sample["utterance"] == "What is the maximum population of a country in Asia?":
                        sample['query'] = "SELECT max(population) FROM country WHERE Continent  =  \"Asia\""
                    elif sample['query'] == "SELECT * FROM country ORDER BY Population LIMIT 1" and sample["utterance"] == "Which country has the smallest population?":
                        sample['query'] = "SELECT * FROM country ORDER BY Population ASC LIMIT 1"
                    elif sample['query'] == "SELECT Name FROM country ORDER BY Population DESC LIMIT 3" and sample["utterance"] == "Give the names of the 3 countries with the lowest.":
                        sample['query'] = "SELECT Name FROM country ORDER BY Population ASC LIMIT 3"
                    elif sample['query'] == "SELECT Name FROM country WHERE continent  =  \"Europe\" AND Population  =  \"80000\"" and sample["utterance"] == "Of those, what are the names of those that have a population of 80000?":
                        sample['query'] = "SELECT Name FROM country WHERE continent  =  \"Europe\" AND Population  =  80000"
                    elif sample['query'] == "SELECT name FROM city WHERE Population BETWEEN 160000 AND 90000" and sample["utterance"] == "Which of those have a population between 160000 and 900000?":
                        sample['query'] = "SELECT name FROM city WHERE Population BETWEEN 160000 AND 900000"
                    elif sample['query'] == "SELECT production_code ,  channel FROM cartoon ORDER BY original_air_date LIMIT 1" and sample["utterance"] == "tell me the production code and channel of the most recently aired cartoon.":
                        sample['query'] = "SELECT production_code ,  channel FROM cartoon ORDER BY original_air_date DESC LIMIT 1"
                    
                    sample['query'] = sample['query'].replace('> =', '>=').replace('< =', '<=').replace('! =', '!=')
                    tmp.append([
                        sample["query"], db_id
                    ])
                golds.append(
                    tmp
                )

        results = evaluate(golds, pred_filename, self.db_dir, etype, self.kmaps, False, False, False)
        return results
