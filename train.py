from human_detection_model import os, define_paths, train_model, load_and_process_images, save_and_load_np_array, plot_training_history

path_home, path_1, path_0 = define_paths()

person_images = load_and_process_images('data')

save_and_load_np_array(person_images, 'processed_data', 'person_images.npy')

model, history = train_model('processed_data')

plot_training_history(history)

model.save('my_model.h5')