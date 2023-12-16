import os
import tqdm
import torch
import numpy as np
import torch.nn.functional as F
def FragmentAugmentation(data, stretch_factor, scaling_factors, shift_steps, noise_std):
    # Load the data (assuming it's in a PyTorch tensor called data)
    T = data.shape[0]
    # Time stretching/compression
    NEW_T_STRETCH = int(T * stretch_factor)

    stretched_data = F.interpolate(data.unsqueeze(0).transpose(-1, -2), size=NEW_T_STRETCH, mode='linear', align_corners=True).squeeze(0).transpose(-1, -2)
    # Change amplitude
    scaling_factors_tensor = torch.tensor(scaling_factors).view(1, -1)  # Convert to tensor and reshape to 1 x C
    scaled_data = stretched_data * scaling_factors_tensor

    # Time shifting
    shifted_data = torch.roll(scaled_data, shifts=shift_steps, dims=0)

    # Add random noise
    noise = torch.randn_like(shifted_data) * noise_std
    augmented_data = shifted_data + noise

    # Save the augmented data
    return augmented_data


def gaussian_kernel(kernel_size, sigma = 1):
    x = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
    g = torch.exp((-x.pow(2) / (2 * sigma**2)))
    return g / g.sum()


# action 920 subject 819
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--save_path', type=str, default='./syn_data/data')
parser.add_argument('--fragments_path', type=str, default='./syn_data/fragments')
parser.add_argument('--sample_num', type=int, default=6830)
args = parser.parse_args()
syn_save_path = args.save_path
framents_path = args.fragments_path
sample_num = args.sample_num

generate_sample_idx = 0
save_path = framents_path
syn_data_save_path = syn_save_path
os.makedirs(syn_data_save_path, exist_ok=True)
num_fragment = len(os.listdir(save_path))
print('Fragment Num: ', num_fragment)
Samples = sample_num
annotation = {}
import random
for sample_idx in tqdm.tqdm(range(3 * Samples)):
    frament_idx = random.randint(1, num_fragment)
    count = random.randint(5, 250)
    #print(frament_idx)
    frament_numpy = np.load(os.path.join(save_path, str(frament_idx) + '.npy'))
    assert frament_numpy.shape[0] > 0
    frament = torch.FloatTensor(frament_numpy)
    #break
    if count < 15:
        sensor = None
        num_audio_idx = []
        density_map = None
        for c in range(count):
            stretch_factor = random.uniform(0.5, 1.5)
            scaling_factors = [random.uniform(0.5, 1.5) for _ in range(6)]
            shift_steps = random.randint(-15, 15)
            noise_std = random.uniform(0.0, 0.3)
            augmented_frament = FragmentAugmentation(frament,
                                                     stretch_factor,
                                                     scaling_factors,
                                                     shift_steps,
                                                     noise_std, )
            density = gaussian_kernel(augmented_frament.shape[0], 20)
            density = density / density.sum()
            if sensor is None:
                sensor = augmented_frament
                density_map = density
            else:
                sensor = torch.cat([sensor, augmented_frament], dim=0)
                density_map = torch.cat([density_map, density], dim=0)
            if len(num_audio_idx) < 3 :
                num_audio_idx.append(sensor.shape[0] + random.randint(-30, 30))
        assert np.rint(density_map.sum().item()) == count
        if sensor.shape[0] < 1200:
            continue

    elif count < 100:
        ir_prob = random.uniform(0.0, 1)

        if ir_prob < 0.5:
            sensor = None
            num_audio_idx = []
            density_map = None
            for c in range(count):
                stretch_factor = random.uniform(0.5, 1.5)
                scaling_factors = [random.uniform(0.5, 1.5) for _ in range(6)]
                shift_steps = random.randint(-15, 15)
                noise_std = random.uniform(0.0, 0.3)
                augmented_frament = FragmentAugmentation(frament,
                                                         stretch_factor,
                                                         scaling_factors,
                                                         shift_steps,
                                                         noise_std, )
                density = gaussian_kernel(augmented_frament.shape[0], 20)
                density = density / density.sum()
                if sensor is None:
                    sensor = augmented_frament
                    density_map = density
                else:
                    sensor = torch.cat([sensor, augmented_frament], dim=0)
                    density_map = torch.cat([density_map, density], dim=0)
                if len(num_audio_idx) < 3:
                    num_audio_idx.append(sensor.shape[0] + random.randint(-30, 30))
            #print(sensor.shape)
            #print(np.rint(density_map.sum().item()), count)
            assert np.rint(density_map.sum().item()) == count
            if sensor.shape[0] < 1200:
                continue

        else:
            sensor = None
            num_audio_idx = []
            density_map = None
            for c in range(count):
                stretch_factor = random.uniform(0.5, 1.5)
                scaling_factors = [random.uniform(0.5, 1.5) for _ in range(6)]
                shift_steps = random.randint(-15, 15)
                noise_std = random.uniform(0.0, 0.3)
                augmented_frament = FragmentAugmentation(frament,
                                                         stretch_factor,
                                                         scaling_factors,
                                                         shift_steps,
                                                         noise_std, )
                density = gaussian_kernel(augmented_frament.shape[0], 20)
                density = density / density.sum()
                if sensor is None:
                    sensor = augmented_frament
                    density_map = density
                else:
                    sensor = torch.cat([sensor, augmented_frament], dim=0)
                    density_map = torch.cat([density_map, density], dim=0)
                if len(num_audio_idx) < 3:
                    num_audio_idx.append(sensor.shape[0] + random.randint(-30, 30))
            ir_count = random.randint(10, 30)
            # Ir Actions(Noise)
            ir_frament_idx = random.randint(1, num_fragment)
            ir_frament_numpy = np.load(os.path.join(save_path, str(ir_frament_idx) + '.npy'))
            ir_frament = torch.FloatTensor(ir_frament_numpy)
            while ir_frament_idx == frament_idx:
                ir_frament_idx = random.randint(1, num_fragment)
            for c in range(ir_count):
                stretch_factor = random.uniform(0.5, 1.5)
                scaling_factors = [random.uniform(0.5, 1.5) for _ in range(6)]
                shift_steps = random.randint(-15, 15)
                noise_std = random.uniform(0.0, 0.3)
                augmented_frament = FragmentAugmentation(ir_frament,
                                                         stretch_factor,
                                                         scaling_factors,
                                                         shift_steps,
                                                         noise_std, )
                density = torch.zeros(augmented_frament.shape[0])
                sensor = torch.cat([sensor, augmented_frament], dim=0)
                density_map = torch.cat([density_map, density], dim=0)
            assert np.rint(density_map.sum().item()) == count
            if sensor.shape[0] < 1200:
                continue

    else:
        count_one = random.randint(int(count * 0.2), int(count * 0.8))
        count_two = count - count_one
        ir_count = random.randint(30, 60)
        sensor = None
        num_audio_idx = []
        density_map = None
        for c in range(count_one):
            stretch_factor = random.uniform(0.5, 1.5)
            scaling_factors = [random.uniform(0.5, 1.5) for _ in range(6)]
            shift_steps = random.randint(-15, 15)
            noise_std = random.uniform(0.0, 0.3)
            augmented_frament = FragmentAugmentation(frament,
                                                     stretch_factor,
                                                     scaling_factors,
                                                     shift_steps,
                                                     noise_std, )
            density = gaussian_kernel(augmented_frament.shape[0], 20)
            density = density / density.sum()
            if sensor is None:
                sensor = augmented_frament
                density_map = density
            else:
                sensor = torch.cat([sensor, augmented_frament], dim=0)
                density_map = torch.cat([density_map, density], dim=0)
            if len(num_audio_idx) < 3:
                num_audio_idx.append(sensor.shape[0] + random.randint(-30, 30))
        #print('dddd', np.rint(density_map.sum().item()))
        assert np.rint(density_map.sum().item()) == count_one
        # Ir Actions(Noise)
        ir_frament_idx = random.randint(1, num_fragment)
        ir_frament_numpy = np.load(os.path.join(save_path, str(ir_frament_idx) + '.npy'))
        ir_frament = torch.FloatTensor(ir_frament_numpy)
        while ir_frament_idx == frament_idx:
            ir_frament_idx = random.randint(1, num_fragment)
        for c in range(ir_count):
            stretch_factor = random.uniform(0.5, 1.5)
            scaling_factors = [random.uniform(0.5, 1.5) for _ in range(6)]
            shift_steps = random.randint(-15, 15)
            noise_std = random.uniform(0.0, 0.3)
            augmented_frament = FragmentAugmentation(ir_frament,
                                                     stretch_factor,
                                                     scaling_factors,
                                                     shift_steps,
                                                     noise_std, )
            density = torch.zeros(augmented_frament.shape[0])
            sensor = torch.cat([sensor, augmented_frament], dim=0)
            density_map = torch.cat([density_map, density], dim=0)
        #print('dddd', np.rint(density_map.sum().item()))
        assert np.rint(density_map.sum().item()) == count_one
        for c in range(count_two):
            stretch_factor = random.uniform(0.5, 1.5)
            scaling_factors = [random.uniform(0.5, 1.5) for _ in range(6)]
            shift_steps = random.randint(-15, 15)
            noise_std = random.uniform(0.0, 0.3)
            augmented_frament = FragmentAugmentation(frament,
                                                     stretch_factor,
                                                     scaling_factors,
                                                     shift_steps,
                                                     noise_std, )
            density = gaussian_kernel(augmented_frament.shape[0], 20)
            density = density / density.sum()
            sensor = torch.cat([sensor, augmented_frament], dim=0)
            density_map = torch.cat([density_map, density], dim=0)
        #print('ccccc', count)
        #print('dddd', np.rint(density_map.sum().item()))
        assert np.rint(density_map.sum().item()) == count
        if sensor.shape[0] > 40000:
            continue

    num_audio_idx = torch.FloatTensor(num_audio_idx)
    generate_sample_idx += 1
    sample_syn_data_save_path = os.path.join(syn_data_save_path, str(generate_sample_idx))
    os.makedirs(sample_syn_data_save_path, exist_ok=True)
    torch.save(sensor, os.path.join(sample_syn_data_save_path, str(generate_sample_idx) + '_sensor.pt'))
    torch.save(density_map, os.path.join(sample_syn_data_save_path, str(generate_sample_idx) + '_density.pt'))
    torch.save(num_audio_idx, os.path.join(sample_syn_data_save_path, str(generate_sample_idx) + '_exemplar_idx.pt'))
    annotation[generate_sample_idx] = {}
    annotation[generate_sample_idx]['count'] = count
    annotation[generate_sample_idx]['exemplar_idx'] = num_audio_idx.numpy().tolist()
    if generate_sample_idx == Samples:
        break