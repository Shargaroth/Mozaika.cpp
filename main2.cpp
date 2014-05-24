#include <cstdlib>
#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <vector>
#include <dirent.h>
#include <cmath>
#include <time.h>

#define WINDOW_1 "Obraz oryginalny"
#define WINDOW_3 "Mozaika"

const int TILES_X = 30; // ilość kafelków w poziomie
const int TILES_Y = TILES_X; // ilość kafelków w pionie, musi być taka sama jak ilość kafelków w poziomie

/*
 * Ten program:
 * 1. Wczytuje obrazki kafelków i wylicza średni kolor (RGB) każdego z nich
 * 2. Dzieli obraz na małe obszary odpowiadające kolejnym kafelkom
 * 3. Dla każdego z tych obszarów wylicza średni kolor
 * 4. Wybiera kafelek, który jest najbliższy wyliczonemu średniemu kolorowi, i go umieszcza w danym miejscu
 * 
 * Plusy i minusy w stosunku do algorytmu genetycznego:
 * + działa
 * + działa szybko
 * - wynik będzie zawsze ten sam
 * - mniejsza zabawa przy tworzeniu programu (ale też kilkanaściekrotnie mniej poświęconego czasu)
 * - brak kafelków w odbiciu lustrzanym
 * - porównuje średnie kolory obszarów, a nie pixel po pixelu, więc jeżeli mamy obszar, który jest np. z lewej strony biały, a z prawej czarny,
 *   i mamy taki sam dwukolorowy kafelek, to w algorytmie genetycznym ma on dużą szansę bycia dopasowanym na dane miejsce,
 *   podczas gdy w tutaj wstawiony zostanie kafelek, którego średni kolor to szary, czyli taki jak średni kolor obszaru, i to prawdopodobnie
 *   największa wada tego podejścia
 */

void getTiles(std::vector<cv::Mat> &, cv::Size, const char*);
int scalarDiff(cv::Scalar &, cv::Scalar &);

int main(int argc, char* argv[]) {
	srand(time(NULL));

	if (argc < 2) { // czy podano argument przy uruchamianiu programu
		std::cout << "Nie podano obrazu do przetworzenia\n";
		return EXIT_FAILURE;
	}

	cv::Mat pictureOryg; // oryginalny wczytany obraz (format BGR)

	char* filename = argv[1];
	pictureOryg = cv::imread(filename, CV_LOAD_IMAGE_COLOR); // wczytaj podany obrazek

	if (!pictureOryg.data) {
		std::cout << "Błąd odczytu obrazu\n";
		return EXIT_FAILURE;
	}

	int width = pictureOryg.cols, height = pictureOryg.rows;
	cv::Size tileSize(width / TILES_X, height / TILES_Y); // wymiary kafelek

	std::vector<cv::Mat> tiles; // lista plików kafelków - obrazków tworzące mozaikę
	getTiles(tiles, tileSize, "pictures"); // załaduj listę kafelków

	if (tiles.size() < 100) {
		std::cout << "W bibliotece obrazów (katalog ./pictures) musi być minimum 100 plików JPG\n";
		return EXIT_FAILURE;
	}

	std::vector<cv::Scalar> tilesAvgColor; // lista średnich kolorów kafelków, jej indeksy odpowiadają indeksom kafelków z wektora tiles
	tilesAvgColor.reserve(tiles.size());

	for (int i = 0; i < tiles.size(); i++) { // wylicz średni kolor dla każdego kafelka
		tilesAvgColor.push_back(cv::mean(tiles[i]));
	}

	cv::Mat pictureMosaic;
	cv::resize(pictureOryg, pictureMosaic, cv::Size(tileSize.width * TILES_X, tileSize.height * TILES_Y)); // skopiowanie obrazka do nowej matrycy, w razie potrzeby nieco zmniejszonego do wymiaru pełnej wielokrotności kafelków

	for (int i = 0; i < TILES_X * TILES_Y; i++) { // nałóż kolejne kafelki
		int posX = (i % TILES_Y) * tileSize.width;
		int posY = (i / TILES_X) * tileSize.height;

		cv::Rect roi(posX, posY, tileSize.width, tileSize.height);
		cv::Mat tilePlace = pictureMosaic(roi); // obszar mozaiki, na który ma być nałożony kafelek

		cv::Scalar targetAvgColor = cv::mean(tilePlace); // średni kolor obszaru, na który ma być nałożony kafelek

		int bestTileNumber = 0;
		int bestDiff = scalarDiff(targetAvgColor, tilesAvgColor[0]);

		for (int j = 1; j < tiles.size(); j++) { // poszukiwanie najlepszego kafelka (najmniej różniącego się od docelowego obszaru)
			int diff = scalarDiff(targetAvgColor, tilesAvgColor[j]);
			if (diff < bestDiff) {
				bestDiff = diff;
				bestTileNumber = j;
			}
		}

		tiles[bestTileNumber].copyTo(tilePlace); // nałóż wybrany kafelek
	}

	cv::namedWindow(WINDOW_1, CV_WINDOW_KEEPRATIO); // okno oryginalnego obrazu
	cv::namedWindow(WINDOW_3, CV_WINDOW_KEEPRATIO); // okno mozaiki

	cv::imshow(WINDOW_1, pictureOryg); // pokaż oryginalny obrazek
	cv::imshow(WINDOW_3, pictureMosaic); // pokaż mozaikę

	int key;
	do {
		key = cv::waitKey(0); // czekaj na naciśnięcie ESC
	} while (key != 27);

	cv::destroyAllWindows(); // zamknij wszystkie okienka

	return EXIT_SUCCESS;
}

// Tworzy tablicę kafelków określonego rozmiaru na podstawie obrazów z podanego katalogu.

void getTiles(std::vector<cv::Mat> &tiles, cv::Size tileSize, const char* directory) {
	DIR *dir;
	struct dirent *ent;

	if ((dir = opendir(directory)) != NULL) {
		while ((ent = readdir(dir)) != NULL) {
			std::string fn = ent->d_name; // pobierz nazwę pliku
			if (fn.substr(fn.find_last_of(".") + 1) == "jpg") { // czy to plik .jpg
				fn = "pictures/" + fn;

				cv::Mat tile = cv::imread(fn, CV_LOAD_IMAGE_COLOR);
				cv::resize(tile, tile, tileSize);
				tiles.push_back(tile); // dodaj do tablicy kafelków
			}
		}
		closedir(dir);
	} else {
		std::cout << "Błąd odczytu biblioteki obrazów\n";
	}
}

// Oblicza różnicę między scalarami - sumę różnic wartości RGB

int scalarDiff(cv::Scalar &s1, cv::Scalar &s2) {
	int diff = 0;

	for (int i = 0; i < 3; i++) {
		diff += std::abs(s1[i] - s2[i]);
	}

	return diff;
}
