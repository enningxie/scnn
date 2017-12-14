import datetime
import pickle
import lasagne


# Log details
def log(f, txt, do_print=1):
    txt = str(datetime.datetime.now()) + ': ' + txt
    if do_print == 1:
        print(txt)
    f.write(txt + '\n')


# Load saved network weights
def load_nets(files, networks):
    print('Loading nets...')
    for file_name, net in zip(files, networks):
        fp = open(file_name, 'rb')
        params = pickle.load(fp, encoding='iso-8859-1')
        lasagne.layers.set_all_param_values(net.output_layer, params)
        print(' >', net.name)
        fp.close()
        print(' > Done.')


# Save network weights
def save_nets(files, networks):
    print('Saving nets...')
    for file_name, net in zip(files, networks):
        fp = open(file_name, 'wb')
        params = lasagne.layers.get_all_param_values(net.output_layer)
        pickle.dump(params, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print(' >', net.name)
        fp.close()
    print(' > Done.')