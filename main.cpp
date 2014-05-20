#include <cstdlib>
#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <vector>
#include <dirent.h>
#include <string>
#include <cmath>
#include <algorithm>

#define WINDOW_1 "Obraz oryginalny"
#define WINDOW_2 "Mozaika poczatkowa"
#define WINDOW_3 "Mozaika wynikowa"

const int TILES_X = 30; // ilość kafelków w poziomie
const int TILES_Y = TILES_X; // ilość kafelków w pionie, musi być taka sama jak ilość kafelków w poziomie

const int POP_SIZE = 300; // liczba osobników w każdej populacji
const int GEN_NUMBER = 150; // ilość pokoleń

struct specimen { // pojedynczy osobnik populacji
	std::vector<int> v; // tablica kolejności kafelków dla osobnika (chromosom)
	cv::Mat m; // macierz wygenerowana na podstawie wektora v
	int fitness; // współczynnik przystosowania - jak bardzo się różni od oryginału; czym mniej tym lepiej
};

void getTiles(std::vector<cv::Mat> &, cv::Size, const char*);
void putTileOnMosaic(cv::Mat, cv::Mat, int);
void initPopulation(std::vector<specimen> &, cv::Mat, cv::Size, std::vector<cv::Mat> &);
int calculateFitness(cv::Mat, cv::Mat);
int tournament(std::vector<specimen> &);
specimen reproduce(std::vector<specimen> &, std::vector<cv::Mat> &, cv::Mat);
void nextGeneration(std::vector<specimen> &, std::vector<specimen> &, std::vector<cv::Mat> &, cv::Mat);

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

	std::vector<specimen> specimens; // tablica osobników
	initPopulation(specimens, pictureOryg, tileSize, tiles); // stwórz początkową populację

	int bestSpecimen = 0;
	for (int i = 1; i < specimens.size(); i++) {
		if (specimens[i].fitness < specimens[i].fitness) {
			bestSpecimen = i;
		}
	}
	std::cout << "Stworzono pokolenie: 0; Najlepszy fitness: " << specimens[bestSpecimen].fitness << "\n";
	
	cv::Mat pictureRandomMosaic = specimens[bestSpecimen].m; // zapamiętaj mozaikę najlepszego osobnika z zerowego pokolenia

	int lastBestFitness = 0;
	for (int i = 1; i <= GEN_NUMBER; i++) {
		nextGeneration(specimens, specimens, tiles, pictureOryg);

		int bestFitness = specimens[0].fitness;
		for (int j = 1; j < specimens.size(); j++) {
			if (specimens[j].fitness < bestFitness) {
				bestFitness = specimens[j].fitness;
			}
		}
		std::cout << "Stworzono pokolenie: " << i << "; Najlepszy fitness: " << bestFitness << "\n";

		if (bestFitness == lastBestFitness) { // ponieważ nasza populacja stała się zbieżna i nie jest dostatecznie różnorodna by dalej coś wyszło
			std::cout << "W dwóch pokoleniach pod rząd wystąpił ten sam najlepszy współczynnik fitness, program nie osiągnie już dużo lepszych rezultatów przez zbieżność osobników, przerwanie wykonania\n";
			break;
		}
		lastBestFitness = bestFitness;
	}

	bestSpecimen = 0;
	for (int i = 1; i < specimens.size(); i++) {
		if (specimens[i].fitness < specimens[i].fitness) {
			bestSpecimen = i;
		}
	}
	
	cv::Mat pictureMosaic = specimens[bestSpecimen].m; // pokaż mozaikę najlepszego osobnika ostatniego pokolenia

	cv::namedWindow(WINDOW_1, CV_WINDOW_KEEPRATIO); // okno oryginalnego obrazu
	cv::namedWindow(WINDOW_2, CV_WINDOW_KEEPRATIO); // okno początkowej mozaiki
	cv::namedWindow(WINDOW_3, CV_WINDOW_KEEPRATIO); // okno wynikowej mozaiki

	cv::imshow(WINDOW_1, pictureOryg); // pokaż oryginalny obrazek
	cv::imshow(WINDOW_2, pictureRandomMosaic); // pokaż początkową mozaikę
	cv::imshow(WINDOW_3, pictureMosaic); // pokaż wynikową mozaikę

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

// Umieszcza kafelek na matrycy mozaiki. Pozycja liczona jest od lewej do prawej od góry do dołu, max pozycja = TILES_X*TILES_Y.
void putTileOnMosaic(cv::Mat mosaic, cv::Mat tile, int position) {
	int posX = (position % TILES_Y) * tile.cols;
	int posY = (position / TILES_X) * tile.rows;

	cv::Rect roi(posX, posY, tile.cols, tile.rows);
	cv::Mat place = mosaic(roi);
	tile.copyTo(place);
	//tile.copyTo(mosaic(roi));
}

// Tworzy początkową populację z losowymi układami kafelków.
void initPopulation(std::vector<specimen> &specimens, cv::Mat pictureOryg, cv::Size tileSize, std::vector<cv::Mat> &tiles) {
	specimens.resize(POP_SIZE); // nadaj rozmiar tablicy na ilość osobników w populacji (tworzy puste osobniki)

	for (int i = 0; i < POP_SIZE; i++) { // stwórz początkowe osobniki
		specimens[i].m = cv::Mat(tileSize.height * TILES_Y, tileSize.width * TILES_X, CV_8UC3); // stwórz osobnikowi pustą macierz
		specimens[i].v.resize(TILES_X * TILES_Y); // nadaj rozmiar tablicy kafelków osobnika równy ilości kafelków
		for (int j = 0; j < TILES_X * TILES_Y; j++) { // stwórz osobnikowi losową tablice kafelków i wrzuć je na macierz
			specimens[i].v[j] = rand() % tiles.size();
			putTileOnMosaic(specimens[i].m, tiles.at(specimens[i].v[j]), j);
		}
		specimens[i].fitness = calculateFitness(pictureOryg, specimens[i].m); // wylicz i zapisz fitness osobnika
	}
}

// Wylicza przystosowanie danej mozaiki - jak bardzo jej kolory różnią się od oryginalnych. Czym mniej tym lepiej.
int calculateFitness(cv::Mat matA, cv::Mat matB) {
	int diff = 0; // różnica kolorów

	for (int i = 0; i < matB.rows; i++) {
		for (int j = 0; j < matB.cols; j++) {
			for (int k = 0; k < 3; k++) {
				diff += std::abs(matA.at<cv::Vec3b>(i, j)[k] - matB.at<cv::Vec3b>(i, j)[k]); // dodaj różnicę kolorów na tym pixelu do sumy
			}
		}
	}

	return diff;
}

// Wybiera osobnika metodą selekcji turniejowej (losuje kilku osobników z populacji i wybiera najlepszego z nich).
int tournament(std::vector<specimen> &specimens) {
	std::vector<int> selectedNumbers; // tablica numerów osobników wybranych do turnieju - bez powtórzeń

	do { // losowanie numerów osobników bez powtórzeń
		int i = rand() % POP_SIZE;
		if (std::find(selectedNumbers.begin(), selectedNumbers.end(), i) == selectedNumbers.end()) { // jeśli tablica numerów nie zawiera wylosowanej liczby
			selectedNumbers.push_back(i);
		}
	} while (selectedNumbers.size() < POP_SIZE / 10); // turniej na 1/10 losowo wybranej części populacji

	int bestSpecimen = selectedNumbers[0]; // najlepszy osobnik wybrany do turnieju - zwyciężca

	for (int i = 1; i < selectedNumbers.size(); i++) { // wybór zwyciężcy turnieju
		if (specimens[selectedNumbers[i]].fitness < specimens[bestSpecimen].fitness) {
			bestSpecimen = selectedNumbers[i];
		}
	}

	return bestSpecimen;
}

// Tworzy nowego osobnika metodą podwójnej selekcji turniejowej, krzyżowania i mutacji.
specimen reproduce(std::vector<specimen> &specimens, std::vector<cv::Mat> &tiles, cv::Mat pictureOryg) {
	int mother = tournament(specimens); // wybierz matkę selekcją turniejową
	int father;
	specimen child;
	
	do { // wybierz ojca selekcją turniejową, ale innego osobnika niż matka
		father = tournament(specimens);
	} while (father == mother);

	std::vector<int> cuts; // miejsca cięć chromosomów
	for (int i = 0; i < specimens[mother].v.size() / 5; i++) { // wylosuj miejsca cięć - ilość cięć = 1/5 dł. chromosomu
		cuts.push_back(rand() % specimens[mother].v.size());
	}
	cuts.push_back(specimens[mother].v.size()); // dodaj cięcie na końcu, aby zawsze tablica dziecka była uzupełniona do końca
	std::sort(cuts.begin(), cuts.end()); // posortuj listę miejsc cięć

	child.v.resize(specimens[mother].v.size());

	int last = 0; // miejsce poprzedniego cięcia, czyli odkąd zacząć doklejanie kolejnego fragmentu chromosomu
	for (int i = 0; i < cuts.size(); i++) { // stwórz chromosom krzyżując chromosomy rodziców
		int parent = (i % 2 == 1) ? father : mother; // na zmianę bierz kawałek chromosomu ojca i matki

		std::copy(specimens[parent].v.begin() + last, specimens[parent].v.begin() + cuts[i], child.v.begin() + last);
		last = cuts[i];
	}
	
	if (rand() % 50 == 0) { // mutacja z prawdopodobieństwem 2%
		for (int i = 0; i < child.v.size() / 100; i++) { // zamień 1/100 kafelków na losowe
			child.v[rand() % child.v.size()] = rand() % tiles.size();
		}
	}

	child.m = cv::Mat(specimens[mother].m.rows, specimens[mother].m.cols, CV_8UC3);
	for (int i = 0; i < child.v.size(); i++) { // stwórz matrycę obrazu na podstawie chromosomu
		putTileOnMosaic(child.m, tiles.at(child.v[i]), i);
	}

	child.fitness = calculateFitness(pictureOryg, child.m); // wylicz funkcję przystosowania dziecka
	
	return child;
}

// Tworzy nowe pokolenie.
void nextGeneration(std::vector<specimen> &oldGeneration, std::vector<specimen> &newGeneration, std::vector<cv::Mat> &tiles, cv::Mat pictureOryg) {
	std::vector<specimen> tempGeneration; // tymczasowa tablica nowych osobników, ponieważ oldGeneration i newGeneration może wskazywać na ten sam wektor
	tempGeneration.resize(POP_SIZE);

	for (int i = 0; i < POP_SIZE; i++) {
		tempGeneration[i] = reproduce(oldGeneration, tiles, pictureOryg);
	}

	newGeneration = tempGeneration;
}
