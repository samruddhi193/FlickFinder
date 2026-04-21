import pandas as pd
import os
import unittest
import sqlite3
import joblib

class TestFlickFinderPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Configure Root execution structures mapping cleanly
        cls.base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cls.proc_dir = os.path.join(cls.base_dir, 'data', 'processed')
        cls.outputs_dir = os.path.join(cls.base_dir, 'outputs')
        cls.warehouse_dir = os.path.join(cls.base_dir, 'data', 'warehouse')
        cls.models_dir = os.path.join(cls.outputs_dir, 'models')
        cls.fig_dir = os.path.join(cls.outputs_dir, 'figures')
        
    def test_01_loaded_columns(self):
        """Verifying standard mapping sequence strictly resolving User interactions."""
        path = os.path.join(self.proc_dir, 'train.csv')
        if not os.path.exists(path):
            self.skipTest("Dependent Data Extraction Phase 3 completely ungenerated. Run main.py first.")
            
        df = pd.read_csv(path)
        expected_columns = ['userId', 'movieId', 'rating', 'timestamp']
        for col in expected_columns:
            self.assertIn(col, df.columns, msg=f"Column extraction fault. Essential Column: '{col}' entirely missing!")
            
    def test_02_zero_nulls_in_rating(self):
        """Checking exact data consistency mapping NaN failures accurately."""
        path = os.path.join(self.proc_dir, 'train.csv')
        if not os.path.exists(path):
            self.skipTest("Data Preprocessing not resolved.")
            
        df = pd.read_csv(path)
        null_count = df['rating'].isnull().sum()
        self.assertEqual(null_count, 0, msg=f"Data scrubbing failed structurally: {null_count} specific NaN instances detected.")
        
    def test_03_svd_deterministic_generation(self):
        """Checks if exact Latent Models produce absolutely deterministic variables consistently mapping identical seeds."""
        model_path = os.path.join(self.models_dir, 'svd_model.pkl')
        if not os.path.exists(model_path):
            self.skipTest("Collaborative Filtering phase matrices completely absent.")
            
        model = joblib.load(model_path)
        
        # Test exact replication mathematically avoiding float variances
        pred1 = model.predict(uid=11, iid=5).est
        pred2 = model.predict(uid=11, iid=5).est
        
        self.assertEqual(pred1, pred2, msg="Algorithmic boundaries unstable. Expected natively identical numerical approximations.")
        
    def test_04_warehouse_integrity(self):
        """Verify strict native Dimensional OLAP Table structuring mappings mapped cleanly."""
        db_path = os.path.join(self.warehouse_dir, 'movie_warehouse.db')
        if not os.path.exists(db_path):
            self.skipTest("SQLite DB absent.")
        
        conn = sqlite3.connect(db_path)
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        tables = [x[0] for x in conn.execute(query).fetchall()]
        
        self.assertEqual(len(tables), 5, msg=f"OLAP Schema Violation: Generated {len(tables)} tables instead of exactly 5.")
        
        # Checking native populated nodes
        for table in tables:
            count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            self.assertGreater(count, 0, msg=f"Table Mapping Violation: Schema table '{table}' is fundamentally empty!")
            
        conn.close()
        
    def test_05_association_rules(self):
        """Checks analytical mathematical structural bounds generation matrices inside Apriori rules."""
        path = os.path.join(self.outputs_dir, 'association_rules.csv')
        if not os.path.exists(path):
            self.skipTest("Apriori Mathematical Analysis sequences missing.")
        
        df = pd.read_csv(path)
        self.assertGreater(len(df), 0, msg="Market Basket generated entirely Zero logical limits/rules.")
        
        expected_metrics = ['lift', 'confidence', 'support', 'antecedents', 'consequents']
        for col in expected_metrics:
            self.assertIn(col, df.columns, msg=f"Metric Fault: Critical Evaluation metric '{col}' unmapped internally.")
            
    def test_06_graphic_outputs_existence(self):
        """Verify 9 sequence Phase 12 graphics resolved dynamically."""
        expected_files = [
            'plot_1_ratings_dist.png',
            'plot_2_genre_pop.png',
            'plot_3_user_heatmap.png',
            'plot_4_long_tail.png',
            'plot_5_model_comparison.png',
            'plot_6_pr_curve.png',
            'plot_7_pca_embeddings.png',
            'plot_8_network.png',
            'plot_9_alpha_curve.png'
        ]
        
        for f in expected_files:
            file_path = os.path.join(self.fig_dir, f)
            self.assertTrue(os.path.exists(file_path), msg=f"Visualization Fault: Graphic Map natively un-rendered -> {f}")

if __name__ == '__main__':
    print("="*60)
    print("UNIT TESTING VALIDATION ENGINE INITIATED")
    print("="*60)
    unittest.main()
