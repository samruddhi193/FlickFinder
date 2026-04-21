import os
import yaml
import logging
import importlib
import sys

# Integrate the local SRC directory into PATH logically preventing any strict dependency injection bugs
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | [%(levelname)s] | %(message)s',
    datefmt='%H:%M:%S'
)

def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def execute_phase(module_name):
    logging.info(f"\n{'='*70}\n🚀 INITIATING PHASE: {module_name}\n{'='*70}")
    try:
        # Import the script sequentially matching pipeline steps
        mod = importlib.import_module(module_name)
        
        # Pull the executable 'run' method properly mapped across all internal nodes
        execution_engine = getattr(mod, 'run', getattr(mod, 'main', None))
        
        if execution_engine:
            execution_engine()
            logging.info(f"✅ Executed Successfully -> [{module_name}]")
        else:
            logging.error(f"❌ MODULE ERROR: {module_name} lacks mapping to 'run()' or 'main()' callable sequences.")
            raise AttributeError("Execution sequence function not found inside Target Module.")
            
    except Exception as e:
        logging.error(f"🚨 CRITICAL CASCADING FAILURE IN {module_name} 🚨", exc_info=True)
        logging.warning("System gracefully catching sequence failure. Pipeline proceeding to next Node theoretically, but dependent structures may fault.")
        # We do not exit() cleanly so logging keeps capturing other errors per requirements

def main():
    config = load_config()
    
    # Safely extract User PRN Config or fallback natively globally 
    prn = config.get("PRN_NUMBER", "PRN_PLACEHOLDER [Assign inside config.yaml]")
    
    print("\n" + "█"*75)
    print("🎬 FLICKFINDER DATA SCIENCE PIPELINE | CAPSTONE EXECUTION ENGINE")
    print(f"🎓 DESIGN ENGINEER: {prn}")
    print("█"*75)
    
    # Establish Sequence Routing (Mock is used to bypass TMDB limits realistically on local setups)
    pipeline_nodes = [
        'src.mock_data_generator',    # Phase 1
        'src.03_preprocess',          # Phase 3 
        'src.04_feature_engineering', # Phase 4
        'src.05_build_warehouse',     # Phase 5
        'src.06_olap_queries',        # Phase 6
        'src.07_collaborative_filtering', # Phase 7
        'src.08_content_based',       # Phase 8
        'src.09_association_rules',   # Phase 9
        'src.10_hybrid_model',        # Phase 10
        'src.11_evaluation',          # Phase 11
        'src.12_visualizations'       # Phase 12
    ]
    
    for phase_schema in pipeline_nodes:
        execute_phase(phase_schema)
        
    print("\n" + "█"*75)
    print("🏁 MASTER PIPELINE TERMINATION COMPLETE. 🏁")
    print("✅ Verify outputs/ and models/ matrices safely.")
    print("█"*75 + "\n")

if __name__ == '__main__':
    main()
