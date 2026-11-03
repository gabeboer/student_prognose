# student_prognose_arima_xgboost.R
# In dit script wordt een studentprognose uitgevoerd op dezelfde wijze als NPULS
# dit doet. Het geeft de vrijheid om zelf de modellen te optimaliseren.
# Input: werkelijke inschrijvingen van 2020-2026 en de Studielink telbestanden

# Stap 1: met een ARIMA (hier gaan we in een later stadium op varieren) de missende
# toekomstige weken voorspellen
# Stap 2: de weken kantelen naar 'x-variabelen' en in een ML algoritme gooien.
# Hier valt al helemaal goed te varieren.

# Lees de packages in en zet gebruiker en wd om de bestanden in te kunnen lezen
library(janitor)
library(tidyr)
library(tidyverse)
library(forecast)
library(purrr)
library(randomForest)
library(caret)
library(rpart)
library(rpart.plot)
library(writexl)

gebruiker <- Sys.getenv("USERNAME")

# input bestanden zijn te vinden op
pad_output <- file.path(
  "C:/Users",
  gebruiker,
  "OneDrive - Universiteit Leiden",
  "IM-BI - R datastraat",
  "R output bestanden"
)

# de telbestanden compleet
isatcodes <- c("50205",
               "50206",
               "50207",
               "55009",
               "55010",
               "56860",
               "56978",
               "56980",
               "60200",
               "60202",
               "60205",
               "60206",
               "60207",
               "60300",
               "60957",
               "60965",
               "65003",
               "66286",
               "66857",
               "66860",
               "66980",
               "50300",
               "65029",
               "50474")


telbestanden_fwn <- readRDS(
  file = file.path(
    pad_output,
    "compleet_telbestand2.rds")) %>% 
  filter(isatcode %in% isatcodes,
         status == "I" | status == "V") %>% 
  # herkomst onbekend op de 'niet-EER' gooien
  mutate(herkomst = case_when(herkomst == "O" ~ "R",
                              TRUE ~ herkomst)) %>% 
  group_by(collegejaar, volgnummer, isatcode, herkomst, geslacht) %>% 
  summarise(gewogen = sum(gewogen_aantal),
            ongewogen = sum(aantal)) %>% 
  ungroup()

# de werkelijke inschrijvingen compleet. Let op: alleen 1e jaars instroom
instroom_fwn <- readRDS(
  file = file.path(
    pad_output,
    "inschrijvingen.rds"
  )) %>% 
  clean_names("snake") %>% 
  filter(info_provider == "ZDUSS064",
         opleidingsvorm == "voltijd",
         semester == "1") %>% 
  rename(eoi = aantal_1jrs_opl_inst,
         isatcode = marktsegment) %>% 
  filter(isatcode %in% isatcodes) %>% 
  select(-x3, 
         -x5,
         -x7,
         -x9,
         -x13,
         -info_provider)

# en nu voorbereiden voor ARIMA. Op welk niveau worden de lijntjes doorgetrokken

# ==============================================================================
doeljaren <- 2024:2026
sleutelkolommen <- c("isatcode", "herkomst", "geslacht")

# Eén waarde per reeks, collegejaar en volgnummer
bron <- telbestanden_fwn %>%
  group_by(
    across(all_of(sleutelkolommen)),
    collegejaar,
    volgnummer
  ) %>%
  summarise(
    gewogen = sum(gewogen),
    .groups = "drop"
  )

# Beschikbare snapshots, over alle reeksen heen
snapshots <- bron %>%
  distinct(collegejaar, volgnummer)


# Voorspelfunctie voor één reeks, doeljaar en peilmoment
voorspel_reeks <- function(data, peil, doeljaar, historische_jaren) {
  
  if (peil == 52L) {
    return(tibble(
      collegejaar = integer(),
      peil_volgnummer = integer(),
      volgnummer = integer(),
      voorspeld_gewogen = double(),
      ondergrens_95 = double(),
      bovengrens_95 = double()
    ))
  }
  
  tijdas <- bind_rows(
    crossing(
      collegejaar = historische_jaren,
      volgnummer = 1:52
    ),
    tibble(
      collegejaar = doeljaar,
      volgnummer = seq_len(peil)
    )
  ) %>%
    arrange(collegejaar, volgnummer)
  
  historie <- tijdas %>%
    left_join(
      data %>%
        select(collegejaar, volgnummer, gewogen) %>%
        mutate(aanwezig = TRUE),
      by = c("collegejaar", "volgnummer")
    ) %>%
    mutate(
      gewogen = if_else(is.na(aanwezig), 0, gewogen)
    ) %>%
    arrange(collegejaar, volgnummer)
  
  if (any(!is.finite(historie$gewogen))) {
    stop("Een reeks bevat ontbrekende of oneindige gewichten.")
  }
  
  reeks <- ts(
    historie$gewogen,
    start = c(min(historische_jaren), 1),
    frequency = 52
  )
  
  model <- auto.arima(
    reeks,
    seasonal = TRUE,
    stepwise = TRUE,
    approximation = TRUE,
    max.p = 2,
    max.q = 2,
    max.P = 1,
    max.Q = 1
  )
  
  prognose <- forecast(
    model,
    h = 52L - peil,
    level = 95
  )
  
  tibble(
    collegejaar = as.integer(doeljaar),
    peil_volgnummer = as.integer(peil),
    volgnummer = seq.int(peil + 1L, 52L),
    voorspeld_gewogen = as.numeric(prognose$mean),
    ondergrens_95 = as.numeric(prognose$lower[, 1]),
    bovengrens_95 = as.numeric(prognose$upper[, 1])
  )
}

# Buitenste map: doeljaren
prognoses_fwn <- map_dfr(
  doeljaren,
  function(doeljaar) {
    
    message("Doeljaar: ", doeljaar)
    
    historische_jaren <- bron %>%
      filter(collegejaar < doeljaar) %>%
      distinct(collegejaar) %>%
      arrange(collegejaar) %>%
      pull(collegejaar)
    
    # Controle op voldoende en aaneengesloten historische jaren
    if (length(historische_jaren) < 2L) {
      stop("Onvoldoende historische jaren voor doeljaar ", doeljaar)
    }
    
    if (any(diff(historische_jaren) != 1L) ||
        max(historische_jaren) != doeljaar - 1L) {
      stop(
        "Historische jaren moeten aaneengesloten zijn tot ",
        doeljaar - 1L
      )
    }
    
    # Alleen peilmomenten die in de bron beschikbaar zijn
    peilmomenten <- snapshots %>%
      filter(collegejaar == doeljaar,
             volgnummer %in% seq(4L, 48L, by = 4L)) %>%
      arrange(volgnummer) %>%
      pull(volgnummer)
    
    if (length(peilmomenten) == 0L) {
      stop("Geen snapshots beschikbaar voor doeljaar ", doeljaar)
    }
    
    # Historie volledig; doeljaar volledig tot laatste peilmoment
    benodigde_snapshots <- bind_rows(
      crossing(
        collegejaar = historische_jaren,
        volgnummer = 1:52
      ),
      tibble(
        collegejaar = doeljaar,
        volgnummer = seq_len(max(peilmomenten))
      )
    )
    
    ontbrekende_snapshots <- benodigde_snapshots %>%
      anti_join(
        snapshots,
        by = c("collegejaar", "volgnummer")
      )
    
    if (nrow(ontbrekende_snapshots) > 0L) {
      print(ontbrekende_snapshots)
      stop("Ontbrekende snapshots voor doeljaar ", doeljaar)
    }
    
    # Reeksen voorbereiden zonder toekomstige jaren
    reeksen <- bron %>%
      filter(collegejaar %in% c(historische_jaren, doeljaar)) %>%
      group_by(across(all_of(sleutelkolommen))) %>%
      nest() %>%
      ungroup()
    
    # Binnenste map: peilmomenten
    map_dfr(
      peilmomenten,
      function(peil) {
        
        message(
          "Doeljaar: ", doeljaar,
          " | Peilmoment: ", peil,
          " / ", max(peilmomenten)
        )
        
        reeksen %>%
          mutate(
            prognose = map(
              data,
              ~ voorspel_reeks(
                data = .x,
                peil = peil,
                doeljaar = doeljaar,
                historische_jaren = historische_jaren
              ),
              .progress = TRUE
            )
          ) %>%
          select(-data) %>%
          unnest(prognose)
      }
    )
  }
)

# In prognoses_fwn blijven de drie sleutelkolommen behouden. collegejaar geeft 
# het doeljaar aan, peil_volgnummer het voorspelmoment en 
# volgnummer de voorspelde week.

saveRDS(prognoses_fwn,
        file = file.path(
              pad_output,
              "prognoses_fwn.rds"
            ))
# ==============================================================================
# oorspronkelijke data per voorspelling er aan vast knopen om een uitgebreid arima 
# outputfile te maken
prognoses_fwn_52 <- readRDS(file = file.path(
  pad_output,
  "prognoses_fwn_52.rds"))

# # een bestand maken waarin de voorspellingen en de werkelijke telbestanden samenkomen
# combi_bestand <- telbestanden_fwn %>% 
#   select(-ongewogen) %>% 
#   mutate(peil_volgnummer = volgnummer,
#          ondergrens_95 = 0,
#          bovengrens_95 = 0,
#          voorspeld_gewogen = gewogen) %>% 
#   relocate(isatcode,
#            herkomst,
#            geslacht,
#            collegejaar,
#            peil_volgnummer,
#            volgnummer,
#            voorspeld_gewogen,
#            ondergrens_95,
#            bovengrens_95)

sleutel <- c("collegejaar", "isatcode", "herkomst", "geslacht")

# 1. Welke combinaties en peilmomenten zijn voorspeld?
peilmomenten <- prognoses_fwn_52 %>%
  distinct(
    across(all_of(sleutel)),
    peil_volgnummer
  )
# 2. Voor iedere combinatie de bekende weken aanmaken en koppelen
werkelijke_regels <- peilmomenten %>%
  mutate(
    volgnummer = map(peil_volgnummer, ~ seq_len(.x))
  ) %>%
  unnest(volgnummer) %>%
  left_join(
    telbestanden_fwn %>%
      select(all_of(sleutel), volgnummer, gewogen),
    by = c(sleutel, "volgnummer"),
    relationship = "many-to-one"
  ) %>%
  transmute(
    across(all_of(sleutel)),
    peil_volgnummer,
    volgnummer,
    voorspeld_gewogen = gewogen,
    ondergrens_95 = NA_real_,
    bovengrens_95 = NA_real_,
    bron = "werkelijk"
  )
# 3. Werkelijkheid en voorspellingen samenvoegen
aanmeldingen_compleet <- bind_rows(
  werkelijke_regels,
  prognoses_fwn_52 %>%
    filter(volgnummer > peil_volgnummer) %>%
    mutate(bron = "ARIMA")
) %>%
  arrange(
    collegejaar, isatcode, herkomst, geslacht,
    peil_volgnummer, volgnummer
  ) %>% 
  mutate(
    voorspeld_gewogen = if_else(
      bron == "werkelijk" & is.na(voorspeld_gewogen),
      0,
      voorspeld_gewogen
    )
    # ,
    # geslacht = case_when(is.na(geslacht) ~ "O",
    #                      TRUE ~ geslacht)
  ) 
 
# controle voor kantelen
dubbele_cellen <- aanmeldingen_compleet %>%
  count(
    collegejaar, isatcode, herkomst, geslacht,
    peil_volgnummer, volgnummer
  ) %>%
  filter(n > 1)


 # en nu naar een behapbare dataset werken 
input_ml <- aanmeldingen_compleet %>%
  filter(peil_volgnummer > 5 & peil_volgnummer <47) %>% 
  select(
    all_of(sleutel), peil_volgnummer,
    volgnummer, voorspeld_gewogen
  ) %>%
  pivot_wider(
    names_from = volgnummer,
    values_from = voorspeld_gewogen,
    names_prefix = "volgnummer_",
    names_sort = TRUE
  ) %>% 
  select(all_of(sleutel), volgnummer_6:volgnummer_46)
# werkelijke_regels %>%
#   filter(is.na(voorspeld_gewogen))

# # de oorspronkelijke (niet voorspelde) regels bepalen en toevoegen
# extra_regels <- prognoses_fwn_52 %>%
#   distinct(
#     across(all_of(sleutelkolommen)),
#     collegejaar,
#     peil_volgnummer
#   ) %>%
#   mutate(volgnummer = peil_volgnummer) %>% 
#   left_join(telbestanden_fwn, by = c(
#     "isatcode",
#     "herkomst",
#     "geslacht",
#     "volgnummer",
#     "collegejaar"
#   )) %>% 
#   select(-ongewogen) %>% 
#   rename(voorspeld_gewogen = gewogen)

# dit is dan de dataset waarop we een test en train dataset gaan maken en dan echt voorspellen
# maar eerst de werkelijkheid ernaast zetten ......
werkelijkheid_fwn <- instroom_fwn %>% 
  select(inschrijvingsjaar:eoi) %>% 
  filter(opleidingsvorm == "voltijd",
         semester == 1) %>% 
  mutate(herkomst = case_when(eer == "EER" ~ "E",
                              eer == "Ned." ~ "N",
                              TRUE ~ "R"),
         geslacht = case_when(geslacht == "U" ~ "O",
                              geslacht == "F" ~ "V",
                              geslacht == "M" ~ "M",
                              TRUE ~ "K"),
         inschrijvingsjaar = as.numeric(inschrijvingsjaar),
         isatcode = as.numeric(isatcode)) %>% 
  select(-eer) %>% 
  rename(collegejaar = inschrijvingsjaar) %>% 
  group_by(collegejaar,
           isatcode,
           geslacht,
           herkomst) %>% 
  summarise(eoi = sum(eoi)) %>% 
  ungroup()



# input_prog_fwn_52 <- prognoses_fwn_52 %>% 
#   select(-ondergrens_95,
#          -bovengrens_95) %>% 
#   bind_rows(extra_regels) %>% 
#   arrange(isatcode,
#           collegejaar,
#           peil_volgnummer,
#           volgnummer,
#           geslacht,
#           herkomst) %>% 
#   # volgnummer kantelen naar de kolommen
#   pivot_wider(
#     id_cols = c(
#       isatcode, herkomst, geslacht,
#       collegejaar, peil_volgnummer
#     ),
#     names_from   = volgnummer,
#     values_from  = voorspeld_gewogen,
#     names_prefix = "volgnummer_",
#     names_sort   = TRUE
#   ) %>% 
#   # eerste volgnummers mogen eruit, want slechte kwaliteit en de laatste ook
#   select(-volgnummer_1,
#          -volgnummer_2,
#          -volgnummer_3,
#          -volgnummer_4,
#          -volgnummer_5) %>% 
#   select(isatcode:volgnummer_40)



# df klaar voor ML algoritmes
input_prog_fwn_52_met_y <- input_ml %>% 
  left_join(werkelijkheid_fwn, by = c(
    "collegejaar",
    "isatcode",
    "geslacht",
    "herkomst"
  )) %>% 
  # drop het collegejaar. we willen niet dat collegejaar een x-variabele is
  select(-collegejaar,
        ) %>% 
  # filter(geslacht != "K" | !is.na(geslacht)) %>% 
  # werkelijke na waarden veranderen in een 0.
  mutate(geslacht = case_when(is.na(geslacht) ~ "O",
                              TRUE ~ geslacht),
         eoi = case_when(is.na(eoi) ~ 0,
                         TRUE ~ eoi),
         across(starts_with("volgnummer_"), ~ replace_na(.x, 0)))

# test of dataset echt schoon is
# Aantal NA's per kolom
na_overzicht <- input_prog_fwn_52_met_y %>%
  summarise(across(everything(), ~ sum(is.na(.x)))) %>%
  tidyr::pivot_longer(
    everything(),
    names_to = "kolom",
    values_to = "aantal_na"
  ) %>%
  filter(aantal_na > 0)




# een onderscheid maken in train en test
set.seed(407290)

# dataset schudden
hussel <- sample(nrow(input_prog_fwn_52_met_y))

n_train <- round(.7 * nrow(input_prog_fwn_52_met_y)) # om te bepalen hoeveel records 70% is
data_perm <- input_prog_fwn_52_met_y[hussel,]

fwn_train <- data_perm[1:n_train,]
fwn_test <- data_perm[(n_train + 1): nrow(input_prog_fwn_52_met_y),]

#===============================================================================

# random forest als eerste model
n_trees <- 3000

rf_model <- randomForest(eoi ~.,
                         data = fwn_train,
                         importance = TRUE,
                         ntree = n_trees)
print(rf_model)

# het getrainde model draaien op de trainingsdata
uitkomst_train <- predict(rf_model, fwn_train)
uitkomst_train2 <-cbind(uitkomst_train, fwn_train) %>% 
  relocate(uitkomst_train, .after = last_col()) %>% 
  mutate(error = eoi - uitkomst_train,
         abs_error = abs(error))

# en op de testdata
uitkomst_test <- predict(rf_model, fwn_test)
uitkomst_test2 <-cbind(uitkomst_test, fwn_test) %>% 
  relocate(uitkomst_test, .after = last_col()) %>% 
  mutate(error = eoi - uitkomst_test,
         abs_error = abs(error))

write_xlsx(uitkomst_test2, 
           path = "uitkomst_test_random_forest.xlsx")
write_xlsx(uitkomst_train2, 
           path = "uitkomst_train_random_forest.xlsx")

# library(dplyr)
# library(ggplot2)

evaluatie <- fwn_test %>%
  mutate(
    voorspeld = as.numeric(predict(rf_model, newdata = fwn_test)),
    fout = voorspeld - eoi
  )

ggplot(evaluatie, aes(x = eoi, y = voorspeld)) +
  geom_point(alpha = 0.4, colour = "#0072B2") +
  geom_abline(
    intercept = 0, slope = 1,
    linetype = "dashed", colour = "red"
  ) +
  coord_equal() +
  labs(
    title = "Random forest: voorspeld versus werkelijk",
    subtitle = "Achtergehouden testdata; dichter bij de lijn is beter",
    x = "Werkelijke EOI",
    y = "Voorspelde EOI"
  ) +
  theme_minimal()

evaluatie %>%
  summarise(
    MAE  = mean(abs(fout)),
    RMSE = sqrt(mean(fout^2)),
    bias = mean(fout),
    R2   = 1 - sum(fout^2) / sum((eoi - mean(eoi))^2)
  )

randomForest::varImpPlot(
  rf_model,
  type = 1,
  n.var = 20,
  main = "Welke variabelen dragen bij aan de voorspelling?"
)

randomForest::varImpPlot(
  rf_model,
  type = 1,
  scale = FALSE,
  n.var = 20,
  main = "Variabelenbelang: gemiddelde toename in MSE"
)
#===============================================================================
# en nu met een decision tree.
fwn_train_fact <- fwn_train %>% 
  mutate(isatcode = as.character(isatcode))
fwn_test_fact <- fwn_test %>% 
  mutate(isatcode = as.character(isatcode))

tree_model <- rpart(
  eoi ~ .,
  data = fwn_train_fact,
  method = "anova",
  control = rpart.control(
    cp = 0.0001,
    minsplit = 30,
    minbucket = 15,
    maxdepth = 5,
    xval = 200
  )
)

printcp(tree_model)
plotcp(tree_model)

# Kies de eenvoudigste boom binnen één standaardfout
# van de laagste kruisvalidatiefout.
cp_tabel <- tree_model$cptable
beste <- which.min(cp_tabel[, "xerror"])

grens <- cp_tabel[beste, "xerror"] +
  cp_tabel[beste, "xstd"]

gekozen_cp <- max(
  cp_tabel[cp_tabel[, "xerror"] <= grens, "CP"]
)

tree_final <- prune(tree_model, cp = gekozen_cp)

rpart.plot(
  tree_final,
  type = 2,
  extra = 101,
  fallen.leaves = TRUE,
  box.palette = "Blues",
  digits = 2,
  main = "Beslisboom voor EOI"
)

evaluatie_tree <- fwn_test_fact %>%
  mutate(
    voorspeld = as.numeric(
      predict(tree_final, newdata = fwn_test_fact)
    ),
    fout = voorspeld - eoi
  )

evaluatie_tree %>%
  summarise(
    MAE  = mean(abs(fout)),
    RMSE = sqrt(mean(fout^2)),
    bias = mean(fout),
    R2   = 1 - sum(fout^2) / sum((eoi - mean(eoi))^2)
  )

# let op: isatcode wordt nu gezien als getal ipv factor. Geeft wat rare splits.
# Bottom line: leuk om te visualeren, maar mindere voorspeller

#===============================================================================
# en nu xgboost
library(xgboost)

train_x <- fwn_train %>% select(-eoi)
test_x  <- fwn_test  %>% select(-eoi)

# Deze velden zijn categorieën
categorieen <- c("isatcode", "herkomst", "geslacht")

for (kolom in categorieen) {
  train_x[[kolom]] <- as.character(train_x[[kolom]])
  test_x[[kolom]]  <- as.character(test_x[[kolom]])
}

# Codering uitsluitend bepalen op de trainingsdata
encoder <- dummyVars(
  ~ .,
  data = train_x,
  fullRank = FALSE
)

x_train <- as.matrix(predict(encoder, newdata = train_x))
x_test  <- as.matrix(predict(encoder, newdata = test_x))

dtrain <- xgb.DMatrix(x_train, label = fwn_train$eoi)
dtest  <- xgb.DMatrix(x_test,  label = fwn_test$eoi)

parameters <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  eta = 0.05,             # Kleine verbetering per nieuwe boom
  max_depth = 3,          # Ondiepe bomen
  min_child_weight = 5,
  subsample = 0.8,        # Deel van de trainingsregels per ronde
  colsample_bytree = 0.8  # Deel van de variabelen per boom
)

set.seed(407290)

cv <- xgb.cv(
  params = parameters,
  data = dtrain,
  nrounds = 2000,
  nfold = 5,
  stratified = FALSE,
  early_stopping_rounds = 50,
  verbose = 1
)

# Ronde met de laagste gemiddelde validatie-RMSE
beste_ronde <- which.min(cv$evaluation_log$test_rmse_mean)

xgb_model <- xgb.train(
  params = parameters,
  data = dtrain,
  nrounds = beste_ronde,
  verbose = 0
)

evaluatie_xgb <- tibble(
  werkelijk = fwn_test$eoi,
  voorspeld = as.numeric(predict(xgb_model, dtest))
) %>%
  mutate(fout = voorspeld - werkelijk)

evaluatie_xgb %>%
  summarise(
    MAE  = mean(abs(fout)),
    RMSE = sqrt(mean(fout^2)),
    bias = mean(fout),
    R2   = 1 - sum(fout^2) /
      sum((werkelijk - mean(werkelijk))^2)
  )

belang <- xgb.importance(model = xgb_model)

xgb.plot.importance(
  belang,
  top_n = 20,
  main = "Variabelenbelang XGBoost"
)
#===============================================================================
