set -x

downscale=1
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -l|--logdir) logdir="$2"; shift ;;
        -d|--data) data="$2"; shift ;;
        -f|--format) format="$2"; shift ;;
        -q|--quantization) quantization="$2"; shift;;
        -a|--alpha) alpha="$2"; shift;;
        -b|--bound) aabb="$2"; shift;;
        --downscale) downscale="$2"; shift;;
        --update_hash) update_hash="$2"; shift;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

time=$(date "+%Y-%m-%d_%H:%M:%S")

if [ "$quantization" = "MDL" ]; then
  if [ "$alpha" = "True" ]; then
    python ../../main.py ../../data/${data} --workspace ../../workspace/${logdir}/${data}/${time} -O -MDL --data_format ${format} --bound ${aabb} --alpha --downscale ${downscale} --update_hash ${update_hash}
  else
    python ../../main.py ../../data/${data} --workspace ../../workspace/${logdir}/${data}/${time} -O -MDL --data_format ${format} --bound ${aabb} --downscale ${downscale} --update_hash ${update_hash}
  fi
elif [ "$quantization" = 'MGL' ]; then
  if [ "$alpha" = "True" ]; then
    python ../../main.py ../../data/${data} --workspace ../../workspace/${logdir}/${data}/${time} -O -MGL --data_format ${format} --bound ${aabb} --alpha --downscale ${downscale} --update_hash ${update_hash}
  else
    python ../../main.py ../../data/${data} --workspace ../../workspace/${logdir}/${data}/${time} -O -MGL --data_format ${format} --bound ${aabb} --downscale ${downscale} --update_hash ${update_hash}
  fi
fi
