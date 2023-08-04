# echo "$1"
# conda activate dogyu

if [ $# -ge 2 ]; then
    declare -i num_variable=$2
else
    declare -i num_variable=20
fi

# $1 : 1번째 파라미터
#      저장된 모델 이름
#
# $2 : 테스트할 epoch 범위(옵션)
#      0부터 $2까지의 epoch의 test set inference 결과물을 확인

# for i in {0..$num_variable..1}
for ((i = 0; i < $num_variable; i++))
    do 
        echo "$1_res$i"
        python add_spec_generated.py "$1_res$i"
        python test_generated.py "$1_res$i"
    done
