install:
	pip install "pip<24.1" && \
	pip install -r requirements.txt

format:
	black *.py

train:
	python train.py

eval:
	echo "###Training-update summary!" > report.md
	echo "\n## Model Metrics" >> report.md
	cat ../Results/metrics.txt >> report.md
	echo "\n## Model Parameters" >> report.md
	cat ../Results/params.txt >> report.md
	echo "\n## Model Experiments" >> report.md
	cat ../Results/links.txt >> report.md
	echo '\n## Confusion Matrix Plot' >> report.md
	echo '![Confusion Matrix](../Results/model_results.png)' >> report.md
	cml comment create report.md

update-branch:
	git config --global user.name $(USER_NAME)
	git config --global user.email $(USER_EMAIL)
	git commit -am "Update with new results"
	git push --force origin HEAD:project

hf-login:
	git pull origin project
	git switch project
	pip install -U "huggingface_hub[cli]"
	huggingface-cli login --token $(HF) --add-to-git-credential

push-hub:
	huggingface-cli upload kamyk0000/Food11-Classification ../Streamlit  --repo-type=space --commit-message="Sync Streamlit"
	#huggingface-cli upload kamyk0000/Food11-Classification ../Gradio  --repo-type=space --commit-message="Sync Gradio"
	huggingface-cli upload kamyk0000/Food11-Classification ../Model /Model --repo-type=space --commit-message="Sync Model"
	huggingface-cli upload kamyk0000/Food11-Classification ../Results /Metrics --repo-type=space --commit-message="Sync Model"

deploy: hf-login push-hub
