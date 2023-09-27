for directory in $(ls saved); do
  for file in $(ls -t "saved/${directory}/" | tail -n +2); do
    if [[ "${file}" =~ "best" || "${file}" =~ "early-stop" ]]; then
      continue
    fi
    if ! [[ "${file}" =~ "pth" ]]; then
      continue
    fi
    rm "saved/${directory}/${file}"
  done
done
