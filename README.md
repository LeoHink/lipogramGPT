# lipogramGPT
A simple implementation using Karpathy's (2023) nanoGPT and fine-tuning it using PPO.


# Data

To begin first download the tiny Shakespeare Dataset:
```
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```
Then make sure you have the file saved as `input.txt` in the same folder as the rest of the code files. 

# Train

To train baseGPT (Character-Level GPT trained on Shakespeare) run `train.py`.

In `train.py`, you can adjust hyperparameter settings. Including the `eval_interval`. This determines how often training and evaluation loss are printed to the console. More importantly, it also determines how often the weights of the model are saved. They are saved at every evaluation interval which is overkill considering the large files. After running keep only the relevant file. 

- Any suggestions on how to improve setting check-points welcome, simply open an issue!

# langmodel.py

This is the GPT implementation based on Karpathy's (2023) Let's Build GPT lecture which can be accessed [Here}(https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa1RFbVRYWUIwOXV1SG9ZVjFseldwdV9RMXhnd3xBQ3Jtc0ttSnJLcVpDVk55ZVl5bVVGQ0QwUFlyRmRkRk5ydkNQLUdQdGV2MEF0U0RvWkNoTWhBSUlIenQ5SlFfekF3SUtHeVVUemlqWjZnU0drZ0ZVdnk0Nkp3NXctM25vT1JxZm12Y3M1Qk53ekRGNkdwZFYxQQ&q=https%3A%2F%2Fgithub.com%2Fkarpathy%2Fng-video-lecture&v=kCc8FmEb1nY)

- In the file you can adjust hyperparameter settings to cater to the compute available to you. If GPU is available the model will automatically run on the GPU.

# PPO_main.py

This implements the PPO fine-tuning to eliminate U counts. Note that if you want to adjust the reward Threshold this needs to be done manually in `GenEnv.py`.

You can run `PPO_main.py` in the terminal to train. This will be very slow unless you have access to a GPU, or you have significantly reduced the size of the GPT in `langmodel.py`.

# print_outputs.py

Does what it says on the tin. Run this with the appropriate weight (might need to be changed in the file) to generate text with your model's weights. 

# Sorry in Advance

This is quite an ugly implementation to recreate this you will have to tweak the files a bit. Adjusting for example the `u_threshold` in GenEnv. If there are issues please reach out!


