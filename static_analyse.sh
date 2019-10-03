debug_file=.static_analysis.txt
project=po_nrl

# autopep8
echo "running autopep8....";
find $project -type f -name "*.py" | xargs autopep8 --aggressive --aggressive -d > $debug_file;
less $debug_file;
rm $debug_file

# other checkers
for checker in "pylint" "flake8"; do

    echo "running $checker...";
    $checker $project > $debug_file;
    less $debug_file;

done

# mypy
echo "running mypy..."
mypy -p $project > $debug_file;
less $debug_file;

# clean up
rm $debug_file
