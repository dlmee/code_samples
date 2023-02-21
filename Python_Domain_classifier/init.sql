DROP DATABASE IF EXISTS ikatadomain;
CREATE DATABASE ikatadomain;
USE ikatadomain;

CREATE TABLE lexicon (
    id INT NOT NULL AUTO_INCREMENT,
    wform VARCHAR(30) NOT NULL,
    morph VARCHAR(30),
    indexed_sents VARCHAR(800),
    PRIMARY KEY (id)
);

CREATE TABLE vectricon (
    id INT NOT NULL AUTO_INCREMENT,
    wform VARCHAR(30) NOT NULL,
    vector VARCHAR(800),
    PRIMARY KEY (id)
);

CREATE TABLE morphicon (
    id INT NOT NULL AUTO_INCREMENT,
    morpheme VARCHAR(30) NOT NULL,
    words VARCHAR(800),
    PRIMARY KEY (id)
);

CREATE TABLE inflecticon (
    id INT NOT NULL AUTO_INCREMENT,
    morpheme VARCHAR(30) NOT NULL,
    inflections VARCHAR(300),
    PRIMARY KEY (id)
);

CREATE TABLE sentences (
    id INT NOT NULL AUTO_INCREMENT,
    sentence VARCHAR(800),
    PRIMARY KEY (id)
);

CREATE TABLE partsofspeech (
    id INT NOT NULL AUTO_INCREMENT,
    word VARCHAR(30) NOT NULL,
    pos1 VARCHAR(20),
    pos2 VARCHAR(20),
    pos3 VARCHAR(20),
    pos4 VARCHAR(20),
    pos5 VARCHAR(20),
    PRIMARY KEY (id)
);