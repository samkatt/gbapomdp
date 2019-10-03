debug_file=.static_analysis.txt
projects=(po_nrl tests)

# autopep8
for project in "${projects[@]}"; do
    echo "running autopep8 on ${project}....";
    find $project -type f -name "*.py" | xargs autopep8 --aggressive --aggressive -d > $debug_file;
    less $debug_file;
done
rm $debug_file

# mypy
for project in "${projects[@]}"; do
    echo "running mypy on ${project}...."
    mypy -p $project > $debug_file;
    less $debug_file;
done

# other checkers
for checker in "pylint" "flake8"; do

    for project in "${projects[@]}"; do
        echo "running $checker on $project...";
        $checker $project > $debug_file;
        less $debug_file;
    done

done

# clean up
rm $debug_file
