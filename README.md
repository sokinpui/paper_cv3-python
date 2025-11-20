```bash
python app.py \
 -i sample/ \
 --paddle_root ../paddle/PaddleSeg \
 --paddle_config pp_liteseg_fabric_512x512_16k.yml \
 --paddle_model best_model/model.pdparams \
 --paddle_python /home/ubuntu/projects/paddle/.direnv/python-3.12.9/bin/python3

```
