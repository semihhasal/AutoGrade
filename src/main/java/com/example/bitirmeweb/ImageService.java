package com.example.bitirmeweb;

import org.springframework.stereotype.Service;

import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.nio.file.Paths;

@Service
public class ImageService {

    private static final String IMAGES_DIR = "src/main/resources/static/images";

    public List<String> listImages() {
        File folder = new File(IMAGES_DIR);
        File[] listOfFiles = folder.listFiles();
        List<String> fileNames = new ArrayList<>();

        if (listOfFiles != null) {
            for (File file : listOfFiles) {
                if (file.isFile()) {
                    fileNames.add(file.getName());
                }
            }
        }
        return fileNames;
    }
}
