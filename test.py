import argparse

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("filepath", type=str)
	args = parser.parse_args()
	print(args.filepath)


if __name__ == '__main__':
	main()
