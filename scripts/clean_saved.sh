for directory in $(ls saved); do
  for file in $(ls -t "saved/${directory}" | tail -n +2); do
    rm "saved/${directory}/${file}"
  done
done
