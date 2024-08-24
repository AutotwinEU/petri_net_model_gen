import autotwin_gmglib as gmg
# import src.autotwin_pnglib as png
import autotwin_pnglib as png

if __name__ == "__main__":
    # load config
    config_path = 'data/croma-case/config.json'
    config = gmg.load_config(config_path)

    # extract log from skg, reconstruct state, and generate input data for Petri net generation
    png.reconstruct_state(config)
    png.generate_input_data(config['path']['recons_state'], config['path']['input_data'])

    # load data, generate Petri net, and save it
    data = png.load_data(config['path']['input_data'])
    alg = png.Algorithm(data)
    alg.generate_model(data)
    alg.save_model(config['path']['model'])
    alg.show_model(config['path']['model'], engine='dot', add_semantic=True)

    # export discovered Petri net to SKG
    model = alg.load_model(config['path']['model'])
    petri_net_id = alg.export_model(model, config)
    print(f'ID of generated Petri net in SKG: {petri_net_id}')

    print('All done!')
