debug_file=.static_analysis.txt

# autopep8
echo "running autopep8....";
find pobnrl -type f -name "*.py" | xargs autopep8 --aggressive --aggressive -d > $debug_file;
less $debug_file;
rm $debug_file

# other checkers
for checker in "pylint" "flake8" "pyflakes"; do

    echo "running $checker...";
    $checker pobnrl > $debug_file;
    less $debug_file;

done

# mypy
echo "running mypy..."
mypy -p pobnrl > $debug_file;
less $debug_file;

# clean up
rm $debug_file
