#set -x

downscale=1
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -l|--logdir) logdir="$2"; shift ;;
        -d|--data) data="$2"; shift ;;
        -f|--format) format="$2"; shift ;;
        -b|--bound) aabb="$2"; shift;;
        --downscale) downscale="$2"; shift;;
        --log2_hashmap_size) log2_hashmap_size="$2"; shift;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

time=$(date "+%Y-%m-%d_%H:%M:%S")

python ../../main.py ../../data/${data} --workspace ../../workspace/${logdir}/${data}/${time} -O --data_format ${format} --bound ${aabb} --downscale ${downscale} --update_hash 0 --log2_hashmap_size ${log2_hashmap_size}
