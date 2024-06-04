import subprocess
import itertools

def work_on_single_scene(
        lf_name,
        height,
        width,
        channel_list,
        angular_list
):
    subprocess.call(
        [
            "python",
            "inference.py",
            "seed=19491001",
            "result_dir=xxx/results",
            "data_path=xxx/data",
            "input_dir=xxx/input_noise",
            "check_val_every_n_epoch=1",
            "fp16=True",
            "training_epochs=12",
            "lr=0.01",
            "epoch_decay=100",
            f"lf_name={lf_name}",
            f"img_height={height}",
            f"img_width={width}",
            "angular_resolution=9",
            "kernel_size=[3,3,3,3,3]",
            f"channel_number={channel_list}",
            f"angular_number={angular_list}",
            "dilation_list=[1,2,2,2,1]",
            "num_base=-1",
            "num_centroid=256"
        ]
        )
    return


scene_list = [
    ['boxes', 512, 512],
]

architecture_list = [
    [[50] * 5, [1] * 5],
]

config_lists = [
    scene_list,
    architecture_list
]

iter_list = itertools.product(*config_lists)
print(iter_list)

for iter_item in iter_list:
    lf_name = iter_item[0][0]
    height = iter_item[0][1]
    width = iter_item[0][2]

    channel_list = iter_item[1][0]
    angular_list = iter_item[1][1]

    work_on_single_scene(
        lf_name= lf_name,
        height= height,
        width= width,
        channel_list= channel_list,
        angular_list= angular_list
    )


