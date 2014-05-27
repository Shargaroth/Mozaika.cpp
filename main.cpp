#include <cstdlib>
#include <iostream>
#include <cv.h>
#include <highgui.h>
#include <vector>
#include <dirent.h>
#include <string>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <time.h>

#define WINDOW_1 "Obraz oryginalny"
#define WINDOW_2 "Mozaika poczatkowa"
#define WINDOW_3 "Mozaika wynikowa"

const int TILES_X = 30; // ilość kafelków w poziomie
const int TILES_Y = TILES_X; // ilość kafelków w pionie, musi być taka sama jak ilość kafelków w poziomie

int POP_SIZE; // liczba osobników w każdej populacji
int GEN_NUMBER; // ilość pokoleń

int PROB_CROSSING; // prawdopodobieństwo krzyżowania
int PROB_MUTATION; // prawdopodobieństwo mutacji

int TOURNAMENT_SIZE; // rozmiar turnieju

struct specimen { // pojedynczy osobnik populacji
	std::vector<int> v; // tablica kolejności kafelków dla osobnika - pierwsza składowa chromosomu
	std::vector<bool> r; // tablica odbicia lustrzanego kafelków z wektora v (true dla odbicia, false dla oryginalnego obrazka) - druga składowa chromosomu
	cv::Mat m; // macierz wygenerowana na podstawie wektora v
	int fitness; // współczynnik przystosowania - jak bardzo kolory są podobne do oryginalnych; czym więcej tym lepiej
};

void getTiles(std::vector<cv::Mat> &, cv::Size, const char*);
void putTileOnMosaic(cv::Mat &, cv::Mat &, int, bool);
void initPopulation(std::vector<specimen> &, cv::Mat, cv::Size, std::vector<cv::Mat> &);
int calculateFitness(cv::Mat, cv::Mat);
int tournament(std::vector<specimen> &);
std::vector<specimen> reproduce(std::vector<specimen> &, std::vector<cv::Mat> &, cv::Mat);
void nextGeneration(std::vector<specimen> &, std::vector<specimen> &, std::vector<cv::Mat> &, cv::Mat);
int readParameter(std::string, int, bool mustBeEven = false);
int readParameter(std::string, int, int, int);
int readPercent(std::string, int);

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

	std::cout << "Możliwe opcje zatrzymania programu:\n"
			<< "    (1) Po osiągnięciu dużej zbieżności\n"
			<< "    (2) Po określonej ilości pokoleń\n"
			<< "Program zostanie zatrzymany zawsze po osiągnięciu maksymalnej wartości fitness\n\n";
	int stopOption = readParameter("Podaj opcję zatrzymania programu", 1, 1, 2);

	if (stopOption == 2) {
		GEN_NUMBER = readParameter("Podaj ilość pokoleń", 50);
	} else {
		GEN_NUMBER = 2; // aby pętla pokoleń wystartowała
	}

	POP_SIZE = readParameter("Podaj rozmiar populacji", 300, true);
	
	TOURNAMENT_SIZE = POP_SIZE / 10; // domyślny rozmiar turnieju: 1/10 populacji
	TOURNAMENT_SIZE = readPercent("Podaj rozmiar turnieju", TOURNAMENT_SIZE);
	
	PROB_CROSSING = readPercent("Podaj prawdopodobieństwo krzyżowania", 95);
	PROB_MUTATION = readPercent("Podaj prawdopodobieństwo mutacji", 2);

	int maxFitness = width * height * 255 * 3; // najlepszy możliwy fitness = ilość pixeli * 3 kolory RGB (obrazek idealnie taki sam)

	std::cout << "\nNajgorszy możliwy fitness: 0\n"
			<< "Najlepszy możliwy fitness: " << maxFitness << "\n\n";

	std::vector<specimen> specimens; // tablica osobników
	initPopulation(specimens, pictureOryg, tileSize, tiles); // stwórz początkową populację

	int bestSpecimen = 0;
	for (int i = 1; i < specimens.size(); i++) {
		if (specimens[i].fitness > specimens[i].fitness) {
			bestSpecimen = i;
		}
	}
	std::cout << "Stworzono pokolenie: 0; Najlepszy fitness: " << specimens[bestSpecimen].fitness << "\n";

	cv::Mat pictureRandomMosaic = specimens[bestSpecimen].m; // zapamiętaj mozaikę najlepszego osobnika z zerowego pokolenia

	int lastBestFitness = 0;
	int i = 1;
	while (true) {
		if (i > GEN_NUMBER && stopOption == 2) { // stop pętli po ilości pokoleń
			break;
		}

		nextGeneration(specimens, specimens, tiles, pictureOryg);

		int bestFitness = specimens[0].fitness;
		for (int j = 1; j < specimens.size(); j++) {
			if (specimens[j].fitness > bestFitness) {
				bestFitness = specimens[j].fitness;
			}
		}
		std::cout << "Stworzono pokolenie: " << i << "; Najlepszy fitness: " << bestFitness << "\n";

		if (bestFitness == maxFitness) { // stop pętli po osiągnięciu najlepszego możliwego rezultatu (niemal nieprawdopodobne bez specjalnie przygotowanych kafelków)
			std::cout << "\nOsiągnięto osobnika z maksymalną wartością fitness\n\n";
			break;
		}
		if (stopOption == 1 && bestFitness == lastBestFitness) { // stop pętli po osiągnięciu dużej zbieżności
			std::cout << "\nW dwóch pokoleniach pod rząd wystąpił ten sam najlepszy współczynnik fitness, program nie osiągnie już dużo lepszych rezultatów przez zbieżność osobników\n\n";
			break;
		}
		lastBestFitness = bestFitness;

		i++;
	}

	bestSpecimen = 0;
	for (int i = 1; i < specimens.size(); i++) {
		if (specimens[i].fitness > specimens[i].fitness) {
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

void putTileOnMosaic(cv::Mat &mosaic, cv::Mat &tile, int position, bool reflect) {
	int posX = (position % TILES_Y) * tile.cols;
	int posY = (position / TILES_X) * tile.rows;

	cv::Rect roi(posX, posY, tile.cols, tile.rows);
	cv::Mat tilePlace = mosaic(roi);
	tile.copyTo(tilePlace);

	if (reflect) {
		cv::flip(tilePlace, tilePlace, 1); // odbicie lustrzane kafelka
	}
}

// Tworzy początkową populację z losowymi układami kafelków.

void initPopulation(std::vector<specimen> &specimens, cv::Mat pictureOryg, cv::Size tileSize, std::vector<cv::Mat> &tiles) {
	specimens.resize(POP_SIZE); // nadaj rozmiar tablicy na ilość osobników w populacji (tworzy puste osobniki)

	for (int i = 0; i < POP_SIZE; i++) { // ustaw początkowe osobniki
		specimens[i].m = cv::Mat(tileSize.height * TILES_Y, tileSize.width * TILES_X, CV_8UC3); // stwórz osobnikowi pustą macierz
		specimens[i].v.resize(TILES_X * TILES_Y); // nadaj rozmiar tablicy kafelków osobnika równy ilości kafelków
		specimens[i].r.resize(TILES_X * TILES_Y);
		for (int j = 0; j < TILES_X * TILES_Y; j++) { // stwórz kafelki
			specimens[i].v[j] = rand() % tiles.size(); // ustaw losowy kafelek
			specimens[i].r[j] = rand() % 2; // przypisz kafelkowi losową wartość odbicia lustrzanego (true/false)
			putTileOnMosaic(specimens[i].m, tiles.at(specimens[i].v[j]), j, specimens[i].r[j]); // wrzuć kafelek na macierz
		}
		specimens[i].fitness = calculateFitness(pictureOryg, specimens[i].m); // wylicz i zapisz fitness osobnika
	}
}

// Wylicza przystosowanie danej mozaiki - jak bardzo jej kolory są podobne do oryginalnych. Czym więcej tym lepiej.

int calculateFitness(cv::Mat matA, cv::Mat matB) {
	int diff = 0; // podobieństwo kolorów, max = width * height * 255 * 3

	for (int i = 0; i < matB.rows; i++) {
		for (int j = 0; j < matB.cols; j++) {
			for (int k = 0; k < 3; k++) {
				diff += 255 - std::abs(matA.at<cv::Vec3b>(i, j)[k] - matB.at<cv::Vec3b>(i, j)[k]); // dodaj różnicę kolorów na tym pixelu do sumy
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
	} while (selectedNumbers.size() < TOURNAMENT_SIZE);

	int bestSpecimen = selectedNumbers[0]; // najlepszy osobnik wybrany do turnieju - zwyciężca

	for (int i = 1; i < selectedNumbers.size(); i++) { // wybór zwyciężcy turnieju
		if (specimens[selectedNumbers[i]].fitness > specimens[bestSpecimen].fitness) {
			bestSpecimen = selectedNumbers[i];
		}
	}

	return bestSpecimen;
}

// Tworzy dwóch nowych osobników z pary rodziców metodą podwójnej selekcji turniejowej, krzyżowania i mutacji.

std::vector<specimen> reproduce(std::vector<specimen> &specimens, std::vector<cv::Mat> &tiles, cv::Mat pictureOryg) {
	int mother = tournament(specimens); // wybierz matkę selekcją turniejową
	int father;

	std::vector<specimen> children; // tablica dzieci powstałych z krzyżowania
	children.resize(2); // stwórz 2 nowe osobniki w tablicy

	do { // wybierz ojca selekcją turniejową, ale innego osobnika niż matka
		father = tournament(specimens);
	} while (father == mother);

	if (rand() % 100 + 1 <= PROB_CROSSING) { // krzyżowanie z zadanym prawdopodobieństwem
		std::vector<int> cuts; // miejsca cięć chromosomów
		for (int i = 0; i < specimens[mother].v.size() / 5; i++) { // wylosuj miejsca cięć - ilość cięć = 1/5 dł. chromosomu
			cuts.push_back(rand() % specimens[mother].v.size());
		}
		cuts.push_back(specimens[mother].v.size()); // dodaj cięcie na końcu, aby zawsze tablica dziecka była uzupełniona do końca
		std::sort(cuts.begin(), cuts.end()); // posortuj listę miejsc cięć

		children[0].v.reserve(specimens[mother].v.size()); // zarezerwuj miejsce w tablicy, by uniknąć jej zwiększania i kopiowania w trakcie dodawania danych
		children[1].v.reserve(specimens[mother].v.size());

		int last = 0; // miejsce poprzedniego cięcia, czyli odkąd zacząć doklejanie kolejnego fragmentu chromosomu
		for (int i = 0; i < cuts.size(); i++) { // stwórz chromosomy dzieci krzyżując chromosomy rodziców
			int parent1 = (i % 2 == 1) ? father : mother; // na zmianę bierz kawałek chromosomu ojca i matki
			int parent2 = (i % 2 == 1) ? mother : father;

			std::copy(specimens[parent1].v.begin() + last, specimens[parent1].v.begin() + cuts[i], std::back_inserter(children[0].v)); // back_inserter potrzebny z powodu użycia reserve zamiast resize kilka linii wyżej
			std::copy(specimens[parent1].r.begin() + last, specimens[parent1].r.begin() + cuts[i], std::back_inserter(children[0].r));
			
			std::copy(specimens[parent2].v.begin() + last, specimens[parent2].v.begin() + cuts[i], std::back_inserter(children[1].v));
			std::copy(specimens[parent2].r.begin() + last, specimens[parent2].r.begin() + cuts[i], std::back_inserter(children[1].r));
			
			last = cuts[i];
		}
	} else { // brak krzyżowania - potomkowie są tacy sami jak rodzice (chyba że wystąpi mutacja)
		children[0].v = specimens[mother].v;
		children[0].r = specimens[mother].r;
		children[1].v = specimens[father].v;
		children[1].r = specimens[father].r;
	}

	for (int i = 0; i < children.size(); i++) { // akcje wykonywane na każdym dziecku
		specimen * child = &children[i];

		if (rand() % 100 + 1 <= PROB_MUTATION) { // mutacja z zadanym prawdopodobieństwem
			for (int i = 0; i < child->v.size() / 100; i++) { // zamień 1/100 kafelków na losowe
				child->v[rand() % child->v.size()] = rand() % tiles.size();
			}
			for (int i = 0; i < child->r.size() / 100; i++) { // zamień wartość odbicia lustrzanego u 1/100 kafelków na przeciwne
				int j = rand() % child->r.size();
				child->r[j] = !child->r[j];
			}
		}

		child->m = cv::Mat(specimens[mother].m.rows, specimens[mother].m.cols, CV_8UC3);
		for (int i = 0; i < child->v.size(); i++) { // stwórz matrycę obrazu na podstawie chromosomu
			putTileOnMosaic(child->m, tiles.at(child->v[i]), i, child->r[i]);
		}

		child->fitness = calculateFitness(pictureOryg, child->m); // wylicz funkcję przystosowania dziecka
	}

	return children;
}

// Tworzy nowe pokolenie.

void nextGeneration(std::vector<specimen> &oldGeneration, std::vector<specimen> &newGeneration, std::vector<cv::Mat> &tiles, cv::Mat pictureOryg) {
	std::vector<specimen> tempGeneration; // tymczasowa tablica nowych osobników, ponieważ oldGeneration i newGeneration może wskazywać na ten sam wektor
	tempGeneration.reserve(POP_SIZE); // zarezerwuj miejsce w tablicy, by uniknąć jej zwiększania i kopiowania w trakcie dodawania osobników

	for (int i = 0; i < POP_SIZE / 2; i++) {
		std::vector<specimen> children = reproduce(oldGeneration, tiles, pictureOryg); // stwórz 2 nowe osobniki
		tempGeneration.insert(tempGeneration.end(), children.begin(), children.end()); // dopisz osobniki na końcu tablicy
	}

	newGeneration = tempGeneration;
}

// Pobiera parametr od użytkownika, kliknięcie enter pozostawia domyślną wartość parametru. Parametr musi być większy od 0.

int readParameter(std::string msg, int defaultValue, bool mustBeEven) {
	std::string temp;
	bool valid = false;
	int value;

	do {
		std::cout << msg << " [" << defaultValue << "]: ";
		std::getline(std::cin, temp);
		value = std::atoi(temp.c_str());

		if (value < 0) {
			std::cout << "Parametr musi być większy od 0\n";
		} else if (mustBeEven && value % 2 == 1) {
			std::cout << "Parametr musi być podzielny przez 2\n";
		} else {
			if (temp.empty()) {
				value = defaultValue;
			}
			valid = true;
		}
	} while (!valid);

	return value;
}

// Pobiera parametr od użytkownika, kliknięcie enter pozostawia domyślną wartość parametru. Parametr musi być w zakresie [min, max].

int readParameter(std::string msg, int defaultValue, int min, int max) {
	int value;
	bool valid = false;

	do {
		value = readParameter(msg, defaultValue);

		if (value < min || value > max) {
			std::cout << "Parametr musi być w zakresie [" << min << ", " << max << "]\n";
		} else {
			valid = true;
		}
	} while (!valid);

	return value;
}

// Pobiera parametr od użytkownika, kliknięcie enter pozostawia domyślną wartość parametru. Parametr musi być w zakresie [0, 100].

int readPercent(std::string msg, int defaultValue) {
	return readParameter(msg, defaultValue, 0, 100);
}
