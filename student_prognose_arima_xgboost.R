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
      filter(collegejaar == doeljaar) %>%
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
# ==============================================================================
# In prognoses_fwn blijven de drie sleutelkolommen behouden. collegejaar geeft 
# het doeljaar aan, peil_volgnummer het voorspelmoment en 
# volgnummer de voorspelde week.

saveRDS(prognoses_fwn,
        file = file.path(
              pad_output,
              "prognoses_fwn.rds"
            ))

# oorspronkelijke data per voorspelling er aan vast knopen om een uitgebreid arima 
# outputfile te maken

# code volgt nog

# dan weer oppakken / kantelen en klaarmaken voor xgboost