#!/bin/bash

# Create the main directory

# Change to the weave directory
cd weave

# Create directories
mkdir -p core data_sources data_processors llm_interfaces prompt_templates data_generators data_validators plugins/data_augmentation plugins/noise_injection plugins/custom_plugins pipelines config utils examples tests/unit tests/integration tests/end_to_end docs

# Create files
touch core/{__init__.py,framework.py,pipeline.py,data_source.py,data_processor.py,llm_interface.py,prompt_manager.py,data_generator.py,data_validator.py,plugin_manager.py}
touch data_sources/{__init__.py,file_source.py,database_source.py,api_source.py,custom_source.py}
touch data_processors/{__init__.py,text_processor.py,json_processor.py,custom_processor.py}
touch llm_interfaces/{__init__.py,openai_interface.py,huggingface_interface.py,custom_interface.py}
touch prompt_templates/{__init__.py,qa_template.py,summarization_template.py,custom_template.py}
touch data_generators/{__init__.py,qa_generator.py,summarization_generator.py,custom_generator.py}
touch data_validators/{__init__.py,quality_validator.py,format_validator.py,custom_validator.py}
touch plugins/__init__.py
touch pipelines/{__init__.py,qa_pipeline.py,summarization_pipeline.py,custom_pipeline.py}
touch config/{default_config.yaml,advanced_config.yaml}
touch utils/{__init__.py,async_utils.py,logging.py,error_handling.py}
touch examples/{quick_start.py,advanced_usage.py,custom_pipeline.py}
touch docs/{api_reference.md,user_guide.md,developer_guide.md}
touch setup.py requirements.txt README.md
