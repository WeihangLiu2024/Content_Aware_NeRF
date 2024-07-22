
hile [[ "$#" -gt 0 ]]; do
    case $1 in
        -l|--logdir) logdir="$2"; shift ;;
        -d|--data) data="$2"; shift ;;
        -f|--format) format="$2"; shift ;;
        -q|--quantization) quantization='$2'; shift;;
        -a|--alpha) alpha='$2'; shift;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

time=$(date "+%Y-%m-%d_%H:%M:%S")

if [ "$quantization" = "MDL" ]; then
  if [ "$alpha" = "True" ] then
    python main.py data/${data} --workspace workspace/${logdir}/${data}/${time} -O -MDL --data_format ${format} --alpha
  else
    python main.py data/${data} --workspace workspace/${logdir}/${data}/${time} -O -MDL --data_format ${format}
  fi
else if [ "$quantization" = 'MGL' ]; then
  if [ "$alpha" = "True" ] then
    python main.py data/${data} --workspace workspace/CA_final/nerf_synthetic/chair -O -MGL --data_format ${format} --alpha
  else
    python main.py data/${data} --workspace workspace/CA_final/nerf_synthetic/chair -O -MGL --data_format ${format}
  fi
fi
fi