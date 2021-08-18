for (($i=100); $i -lt 501; ($i=$i+100)) {
	python main_sawtooth.py $i
}

for (($i=100); $i -lt 501; ($i=$i+100)) {
	python main_cosine.py $i
}

for (($i=100); $i -lt 501; ($i=$i+100)) {
	python main_sine.py $i
}

for (($i=300); $i -lt 801; ($i=$i+100)) {
	python main_sineprod.py $i
}

for (($i=400); $i -lt 901; ($i=$i+100)) {
	python main_accordian.py $i
}