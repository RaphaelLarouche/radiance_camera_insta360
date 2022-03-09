from processing import ProcessImage

if __name__ == "__main__":
    process = ProcessImage()
    ze_mesh, az_mesh, rad_profile = process.open_radiance_data(path="data/oden-08312018.h5")
    print(az_mesh, '\n', ze_mesh)