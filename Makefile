.PHONY: profile clean static_analyse

test:
	python setup.py test

profile_output = model_based.cprof
profile_results = model_based.res

outputs = $(profile_output) $(profile_results) .static_analysis.txt

profile:
	python -O -m cProfile -o $(profile_output) $(shell which model_based.py) \
		-D tiger --runs 1 --episodes 100 -v 1 \
		--expl 100 --num_sims 4096 --num_part 1024 -B importance_sampling --belief_min 128 \
		--num_pret 4096 --alpha .1 --train on_prior --prior_cert 10000 --num_nets 1 --prior_corr 0 \
		--dropout .5 --online_learning_rate .001 --replay -f $(profile_results)

static_analyse:
	bash ./static_analyse.sh

clean:
	rm -f $(outputs)
