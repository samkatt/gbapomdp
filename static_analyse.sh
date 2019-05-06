# autopep8
autopep8_file=autopep8_debug.txt
echo "running autopep8...." && find pobnrl -type f -name "*.py" | xargs autopep8 --aggressive --aggressive -d >> $autopep8_file && less $autopep8_file && rm $autopep8_file

# other checkers
for checker in "pylint" "flake8" "pyflakes" "mypy"; do

    echo "running $checker..."

    $checker pobnrl >> ${checker}_debug.txt

    less ${checker}_debug.txt && rm ${checker}_debug.txt

done
